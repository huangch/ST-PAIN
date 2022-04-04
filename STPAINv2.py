import numpy as np
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses
from tensorflow.keras import  optimizers
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

def pseudospot_generator(Xs, ys, nmix=5, n_samples=10000):
    nclss=len(set(ys))
    Xs_new, ys_new =[], []
    ys_ = to_categorical(ys)
    for i in range(n_samples):
        yy = np.zeros(nclss)
        fraction = np.random.rand(nmix)
        fraction = fraction/np.sum(fraction)
        fraction = np.reshape(fraction, (nmix,1))
        randindex = np.random.randint(len(Xs), size=nmix)
        ymix = ys_[randindex]
        yy = np.sum(ymix*np.reshape(fraction, (nmix,1)), axis=0)
        XX = Xs[randindex] * fraction
        XX_ = np.sum(XX, axis=0)
        ys_new.append(yy)
        Xs_new.append(XX_)
    Xs_new = np.asarray(Xs_new)
    ys_new = np.asarray(ys_new)
    return Xs_new, ys_new


def build_models(inp_dim, emb_dim, n_cls_source, alpha=2, alpha_lr=10):
    inputs = Input(shape=(inp_dim,)) 
    embeddings = Dense(1024, activation='linear')(inputs)
    embeddings = BatchNormalization()(embeddings)
    embeddings = Activation("elu")(embeddings)  
    embeddings = Dense(emb_dim, activation='linear')(embeddings)
    embeddings = BatchNormalization()(embeddings)
    embeddings = Activation("elu")(embeddings)      

    source_classifier = Dense(n_cls_source, activation='linear', name="source_classifier_1")(embeddings)  
    source_classifier = Activation('softmax', name='source_classifier')(source_classifier)

    exp_domain_classifier = Dense(32, activation='linear', name="exp_domain_classifier_4")(embeddings)
    exp_domain_classifier = BatchNormalization(name="exp_domain_classifier_5")(exp_domain_classifier)
    exp_domain_classifier = Activation("elu", name="exp_domain_classifier_6")(exp_domain_classifier)
    exp_domain_classifier = Dropout(0.5)(exp_domain_classifier)
    exp_domain_classifier = Dense(2, activation='softmax', name="exp_domain_classifier")(exp_domain_classifier)
    
    comb_model = Model(inputs=inputs, outputs=[source_classifier, exp_domain_classifier])
    comb_model.compile(optimizer="Adam",
                       loss={'source_classifier': 'kld', 'exp_domain_classifier': 'categorical_crossentropy'},
                       loss_weights={'source_classifier': 1, 'exp_domain_classifier': alpha}, metrics=['accuracy'], )

    source_classification_model = Model(inputs=inputs, outputs=source_classifier)
    source_classification_model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                                        loss={'source_classifier': 'kld'}, metrics=['mae'], )

    exp_domain_classification_model = Model(inputs=inputs, outputs=exp_domain_classifier)
    
    exp_domain_classification_model.compile(optimizer=optimizers.Adam(learning_rate=alpha_lr*0.001),
                                            loss={'exp_domain_classifier': 'categorical_crossentropy'}, metrics=['accuracy'])
        
    embeddings_model = Model(inputs=inputs, outputs=embeddings)
    embeddings_model.compile(optimizer="Adam",loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    
    return comb_model, source_classification_model, exp_domain_classification_model, embeddings_model


def batch_generator(data, batch_size):
    """Generate batches of data.
    Given a list of numpy data, it iterates over the list and returns batches of the same size
    This
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr
        
        
def train( Xs, ys, Xt,
          emb_dim=2,
          batch_size = 64, 
          n_iterations = 1000,
          alpha=2,
          alpha_lr=10,
          initial_train_epochs=100):
    
    inp_dim = Xs.shape[1]
    ncls_source = ys.shape[1]
    
    model, source_classification_model, exp_domain_classification_model, embeddings_model = \
          build_models(inp_dim, emb_dim, ncls_source, alpha=alpha, alpha_lr = alpha_lr)
          
    source_classification_model.fit(Xs, ys, batch_size= batch_size, epochs=initial_train_epochs)
    print("initial_train_done")
    
    y_adversarial_1 = to_categorical(np.array(([1] * batch_size + [0] * batch_size)))
    y_adversarial_2 = to_categorical(np.array(([0] * batch_size + [1] * batch_size)))

    sample_weights_class = np.array(([1] * batch_size + [0] * batch_size))
    sample_weights_adversarial = np.ones((batch_size * 2,))

    S_batches = batch_generator([Xs, ys], batch_size)
    T_batches = batch_generator([Xt, np.zeros(shape = (len(Xt),2))], batch_size)
    
    for i in range(n_iterations):        
        X0, y0 = next(S_batches)
        X1, y1 = next(T_batches)
        
        X_adv = np.concatenate([X0, X1])
        y_class = np.concatenate([y0, np.zeros_like(y0)])

        adv_weights = []
        for layer in model.layers:
            if (layer.name.startswith("exp_domain_classifier")):
                adv_weights.append(layer.get_weights())
        
        
        # note - even though we save and append weights, the batchnorms moving means and variances
        # are not saved throught this mechanism 
        model.train_on_batch(X_adv, [y_class, y_adversarial_1],
                             sample_weight=[sample_weights_class, sample_weights_adversarial])

        k = 0
        for layer in model.layers:
            if (layer.name.startswith("exp_domain_classifier")):
                layer.set_weights(adv_weights[k])
                k += 1
        
        class_weights = []

        for layer in model.layers:
            if (not layer.name.startswith("exp_domain_classifier")):
                class_weights.append(layer.get_weights())                
        
        exp_domain_classification_model.train_on_batch(X_adv, y_adversarial_2)

        k = 0
        for layer in model.layers:
            if (not layer.name.startswith("exp_domain_classifier")):
                layer.set_weights(class_weights[k])
                k += 1

        if ((i + 1) % 100 == 0):
            # print(i, stats)
            sourceloss, sourceacc = source_classification_model.evaluate(Xs, ys,verbose=0)
            domainloss,domainacc  = exp_domain_classification_model.evaluate(np.concatenate([Xs, Xt]),
                                                                         to_categorical(np.array(([1] * Xs.shape[0] + [0] * Xt.shape[0]))),
                                                                         verbose=0)
            print("Iteration %d, source loss =  %.3f, discriminator acc = %.3f"%(i+1, sourceloss ,domainacc))

    return embeddings_model, source_classification_model 

