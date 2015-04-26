import numpy as np
import scipy as sp
import climin
import save

# Prevent floating point errors (esp nan values) from propagating.
# (The nonexistent word_vecs are set to nan; make sure we don't use them.)
old_settings = np.seterr(all='raise')

# TODO: Adapt for the other problems...

def devise_loss_one_sample(params, image_vec, label, labels_perm, word_vecs,
                           margin=0.1,
                           return_loss=True,
                           return_grad=True,
                           all_terms=False,
                           print_loss=False,
                           only_one_term=True):
    """Compute DeVISE objective and gradient, given a (random) label sequence."""

    if only_one_term and all_terms:
        raise Exception("Contradiction")
    
    word_dim  = word_vecs.shape[1]
    image_dim = np.size(image_vec)

#    print("params shape: ", params.shape)
#    print("word_dim: ", word_dim)
#    print("image_dim: ", image_dim)

    M = np.reshape(params, (word_dim, image_dim))

    Mv = np.dot(M, image_vec.T)
    
    loss = 0.0
    
    # Accumulate the word vectors that contributed error.
    # Then compute the gradient (an outer product) at the end, so
    # we don't have to compute outer products along the way.

    w_label_Mv = np.dot(word_vecs[label,:], Mv)
    sum_w_err  = np.zeros(word_dim)
    
    n_loss = 0
    
    for j in labels_perm:        
#        print("PRE-IF  Debug: j = %d, word_vecs[j,0] = %g" % (j, word_vecs[j,0]))
        if j != label and not np.isnan(word_vecs[j,0]):
#            print("POST-IF Debug: j = %d, word_vecs[j,0] = %g" % (j, word_vecs[j,0]))
            w_j_Mv = np.dot(word_vecs[j,:], Mv)
            loss_j = max(0.0, margin - w_label_Mv + w_j_Mv)

            if loss_j > 0:
                n_loss    += 1
                # We suffered a loss
                loss      += loss_j                                                
                sum_w_err += word_vecs[j,:]
                
                if only_one_term:
                    break                    
            
            # Remove this part for less noisy estimates?
            if loss_j > 1.0 and (not all_terms):
                # We suffered a margin violating loss; finish iteration
                break
                   
                    
    # Since each iteration adds a different number of terms to the loss,
    # scale them.
    
    n_classes = word_vecs.shape[0]
    scale = (n_classes - 1.0) / n_loss if n_loss > 0 else 0
    loss *= scale
                    
    if return_grad:
        grad_flat = scale * np.ravel(
                        np.outer(-n_loss * word_vecs[label,:] + sum_w_err,
                        image_vec)        
                        )
                
    if return_loss and return_grad:
        return (loss, grad_flat)
    elif return_grad:
        return grad_flat
    elif return_loss:
        return loss
    else:
        raise Exception("This code branch should be unreachable!")
    

# Vectorization is worthwhile if Mv takes a lot more time.
# (which it might...)
            
# Vectorized to compute on a minibatch.
#def devise_loss(M, image_vecs, labels, labels_perms):
#    """Compute DeVISE objective and gradient, given a (random) label sequence."""
#    
#    n_minibatch = len(labels)
#    MVs = np.dot(M, image_vecs)
    
#    # The next part is tricky to vectorize because the number of iterations
#    # differs...
#    for n in n_minibatch:
#        for j in labels_perm[n,:]:
#            if j != labels[n]:
#                d = word_vec
            
# Returns an infinite iterator that gives a minibatch of (image_vec, label, labels_perm).

def make_minibatch_iterator(image_vecs, image_labels, word_vecs, n_minibatch=1):
    assert n_minibatch == 1, "Large minibatches not supported."

    n_samples = len(image_labels)
    n_classes = word_vecs.shape[0]
    
    while True:
        rows  = np.random.choice(n_samples, size=n_minibatch, replace=False)
        perms = np.random.permutation(n_classes)
#        perms = np.zeros((n_minibatch, n_classes), dtype="int")
#        for n in range(n_minibatch):
#            perms[n,:] = np.random.permutation(n_classes)                

#        print("make_minibatch_iterator DEBUG: n_samples = %d, rows = %s" % (n_samples, rows))
#        print(np.linalg.norm(image_vecs[rows,:], 1))
        # yield a tuple of (args, kwargs)
        yield ((image_vecs[rows,:], image_labels[rows], perms, word_vecs),
               {"return_loss": False})            

# TODO: Use the actual losses. This is just to verify that algorithm works.
def validation_loss(params, image_vecs, image_labels, word_vecs, validation_inds):
    """Test on only the specified range of entries."""
    loss = 0
    n_classes   = word_vecs.shape[0]
    labels_perm = np.arange(n_classes)

    for n in validation_inds:
        loss += devise_loss_one_sample(params, image_vecs[n,:], image_labels[n], labels_perm, word_vecs,
                                       return_loss=True, return_grad=False,
                                       all_terms=True, only_one_term=False)

    return loss

def run_devise(image_vecs, image_labels, word_vecs, n_epochs, checkpoint_file, iters_per_checkpoint, iters_per_eval, validation_inds, logfile, step_rate=1e-4, decay=0.9, dm_thresh=0.1):
    # TODO:
    n_samples   = len(image_labels)
    n_minibatch = 1
    n_iters     = int(np.ceil(n_epochs * n_samples / n_minibatch))

    word_dim  = word_vecs.shape[1]
    image_dim = image_vecs.shape[1]
    
    # Initialize M
    m_flat = np.random.randn(word_dim * image_dim)    
#    m_flat = np.zeros(word_dim * image_dim)
#    m_flat = np.random.randn(word_dim * image_dim)
    
    # Beware momentum, as it can cause nonconvergence.
    devise_args = make_minibatch_iterator(image_vecs, image_labels, word_vecs, n_minibatch=1)
    #opt = climin.RmsProp(m_flat, devise_loss_one_sample, step_rate=step_rate, decay=decay, args=devise_args)
    opt = climin.GradientDescent(m_flat, devise_loss_one_sample, step_rate=step_rate, momentum=.95, args=devise_args)

    old_m_flat = np.copy(m_flat)
    
    last_validation_loss = np.nan

    lf = open(logfile, 'w')

    for info in opt:
        if info["n_iter"] % iters_per_checkpoint == 0:
            save.save(checkpoint_file, info=info, m_flat=m_flat, last_validation_loss=last_validation_loss)

        # No validation set yet
        if info["n_iter"] % iters_per_eval == 0:
            dm = np.linalg.norm(m_flat - old_m_flat, 1)

            if dm < dm_thresh:
                print("Optimization converged at %d iters: dm < %g." % (info["n_iter"], dm))
                return (M, info)

            old_m_flat = np.copy(m_flat)
            last_validation_loss = validation_loss(m_flat, image_vecs, image_labels, word_vecs, validation_inds)
            print("Iter %d, dM (1-norm) = %g, validation loss = %g" % (info["n_iter"], dm, last_validation_loss))
            lf.write("Iter %d, dM (1-norm) = %g, validation loss = %g\n" % (info["n_iter"], dm, last_validation_loss))
            lf.flush()

        if info["n_iter"] == n_iters:
            M = np.reshape(m_flat, (word_dim, image_dim))
            lf.close()
            return (M, info)

