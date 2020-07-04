from wordcloud import WordCloud, STOPWORDS

# Extracting merchant name and type from maps_data

def extract_maps(maps):
    # Initialize merchant name and type to read value from the lists

    merch_type = []
    merch_name = []
    reviews    = []
    
    for merch in maps:
        rev = []
        try:
            merch_name.append(merch['merchant_name'])
        except:
            merch_name.append('None')

        try:
            #for i in merch['place_results']['type']:           
            merch_type.append(merch['place_results']['type'])
        except:
            merch_type.append('')
        
        try:
            for i in merch['place_results']['user_reviews']['most_relevant']:
                rev.append(i['description'])
        except:
            rev= ''
        reviews.append(rev)
    return merch_name, merch_type, reviews 

# Extracting merchant name and type from search_data

def extract_search(search):
    
    # Initialize snippets to read value from the lists
    
    snippets   = []
    title      = []
    #merch_name = []
    
    for merch in search:
        snip = []
        t = []
        try:
            for i in range(0,4): 
                snip.append(merch['organic_results'][i]['snippet'])
                t.append(merch['organic_results'][i]['title'])
        except:
            snip = ''
            t = ''
        #try:
        #    merch_name.append(merch['merchant_name'])
        #except:
        #    merch_name.append('None')
        snippets.append(snip)
        title.append(t)
    return snippets, title


#def combining_text_wo_reviews(m_name,m_type, reviews,snippet):
#    return m_name + ' ' + m_type + ' ' + snippet

#def combining_text_w_reviews(m_name,m_type, reviews, snippet):
#    return m_name + ' ' + m_type + ' ' + reviews + ' '+ snippet

#def combining_text_title(m_name,m_type, reviews, snippet, title):
#    return m_name + ' ' + m_type  + ' '+ title

def combining_snip_title(m_name,m_type, reviews, snippet, title):
    return m_name + ' ' + m_type  + ' '+ title + ' '+ snippet

#def combining_snip_title_rev(m_name,m_type, reviews, snippet, title):
#    return m_name + ' ' + m_type  + ' '+ title + ' '+ snippet + ' ' + reviews

# Function to extract features from the list and store as string 

def de_list(x):
    words = ''
    if isinstance(x,list):
        for i in x:
            words = words+ ' '+i
    else:
        words = words + x
    return words

# Function to get wordcloud

def get_cloud(category, column, df_merged, STOPWORDS):
    comment_words = '' 
    stopwords = set(STOPWORDS) 

    # iterate through the csv file 
    for val in df_merged[df_merged['category'] == category][column]: 

        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 

        # Converts each token into lowercase 
    #    for i in range(len(tokens)): 
    #        tokens[i] = tokens[i].lower() 

        comment_words += " ".join(tokens)+" "

    wordcloud = WordCloud(width = 500, height = 500, 
                    background_color ='white', 
                    stopwords = stopwords,
                    collocations=False,
                    min_font_size = 10).generate(comment_words) 
    return wordcloud

# Data Preprocessing functions

# Lower case
def text_lowercase(text): 
    return text.lower() 

# Remove numbers 
def remove_numbers(text): 
    result = re.sub(r'\d+', '', text) 
    return result

# remove punctuation 
def remove_punctuation(text): 
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
    return text.translate(translator) 

# remove whitespace from text 
def remove_whitespace(text): 
    return  " ".join(text.split())

# remove stopwords function 
def remove_stopwords(text): 
    words = ''
    #stop_custome = {'india','indian', 'south', 'ltd', 'pvt', 'limit', 'store', 'compani', 'one', 'rate, servic', 'review',
    #               'get','also', 'mumbai', 'delhi'}
    stop_words = set(stopwords.words("english"))
    #stop_words.update(stop_custome)
    #stop_words.update(new_stop)
    word_tokens = word_tokenize(text) 
    for word in word_tokens:
        if word not in stop_words:
            words= words+' '+word
    return words 

# stem words in the list of tokenised words 
def stem_words(text):
    stems= ''
    word_tokens = word_tokenize(text) 
    for word in word_tokens:
        stems = stems + ' ' +stemmer.stem(word) 
    return stems

# transforms data-set to TF-IDF vector

def tfidf_transformation(train,test):
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(min_df=2,max_features  = 5000)
    
    # Fit tfidf vector with training set
    tfidf_fit = tfidf_vectorizer.fit(train)
        
    # transform training and test set on fitted tfidf_vectorizer
    X_train_tfidf = tfidf_fit.transform(train)
    X_test_tfidf = tfidf_fit.transform(test)
    
    return X_train_tfidf,X_test_tfidf

# Function to get results using SVM on TF-IDF 

def get_results(state,data):
    
    result = []
    
    # Running different models for different random state
    
    for i in state:
        # Splitting data to train and test
        X_train, X_test, y_train, y_test = train_test_split(data['snip_title'], data['category'], test_size = 0.20, random_state=i)
        
        # Transform data sets to tfidf vectors
        X_train_tfidf, X_test_tfidf = tfidf_transformation(X_train,X_test)
        
        # Encode Y labels
        Train_Y, Test_Y = labels_getter(y_train,y_test)
        
        # initialize SVM model
        SVM = svm.SVC( kernel='linear', degree=3 , gamma='auto')
        
        # Fit SVM model on training data
        SVM.fit(X_train_tfidf,Train_Y)
        
        # Prediction
        predictions_SVM = SVM.predict(X_test_tfidf)
        
        print(accuracy_score(predictions_SVM, Test_Y)*100)
        
        result.append(accuracy_score(predictions_SVM, Test_Y)*100)
    
    return result


def labels_getter(train, test):
    
    Encoder = LabelEncoder()
    
    # Fit labels based on training data
    label = Encoder.fit(train)
    
    # Transform train and test
    Train_Y = label.transform(train)
    Test_Y  = label.transform(test)
    
    return Train_Y, Test_Y

def deep_model(model, X_train, y_train, X_valid, y_valid, NB_START_EPOCHS, BATCH_SIZE):
    '''
    Function to train a multi-class model. The number of epochs and 
    batch_size are set by the constants at the top of the
    notebook. 
    
    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='adam'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    
    history = model.fit(X_train
                       , y_train
                       , epochs=NB_START_EPOCHS
                       , batch_size=BATCH_SIZE
                       , validation_data=(X_valid, y_valid)
                       , verbose=0)
    return history


def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()

def test_model(model, X_train, y_train, X_test, y_test, epoch_stop, BATCH_SIZE):
    '''
    Function to test the model on new data after training it
    on the full training data with the optimal number of epochs.
    
    Parameters:
        model : trained model
        X_train : training features
        y_train : training target
        X_test : test features
        y_test : test target
        epochs : optimal number of epochs
    Output:
        test accuracy and test loss
    '''
    model.fit(X_train
              , y_train
              , epochs=epoch_stop
              , batch_size=BATCH_SIZE
              , verbose=0)
    results = model.evaluate(X_test, y_test)
    
    return results

def get_model(NB_WORDS,MAX_LEN):
    model = Sequential()
    model.add(Embedding(NB_WORDS, 100, input_length=MAX_LEN))

    model.add(Flatten())
    model.add(Dropout(0.3, input_shape=(14,)))
    model.add(Dense(14, activation='softmax'))
    
    return model


def embedding_result(state, data):
    
    NB_START_EPOCHS = 6  # Number of epochs we usually start to train with
    BATCH_SIZE = 128     # Number of Batch size 
    
    result = []          # To store accuracy
    
    
    for i in state:
        
        # Split data into training and test sets.
        X_train, X_test, y_train, y_test = train_test_split(data['snip_title'], data['category'], test_size = 0.20, random_state=i)
        
        # Encode Y labels.
        Train_Y, Test_Y = labels_getter(y_train,y_test)
        
        # Convert the Encoded labels to categorical format
        y_train_oh = to_categorical(Train_Y)
        y_test_oh = to_categorical(Test_Y)
        
        
        #Initialize and create tokens based on training set
        tk = Tokenizer()
        tk.fit_on_texts(X_train)
        
        # Create sequences based on tokens generated
        X_train_seq = tk.texts_to_sequences(X_train)
        X_test_seq = tk.texts_to_sequences(X_test)
        
        # Maximum Length of row in the test dataset
        seq_lengths = X_train.apply(lambda x: len(x.split(' ')))
        
        MAX_LEN = int(seq_lengths.describe()[-1])
        
        # Pad the dataset to make equal dimension
        X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN , padding = 'post')
        X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding = 'post')
        
        NB_WORDS = len(tk.word_index) +1  # Parameter indicating the number of words we'll put in the dictionary
        
        # get the model
        model= get_model(NB_WORDS,MAX_LEN)
        
        # Split training set to train and validation
        X_train_emb, X_valid_emb, y_train_emb, y_valid_emb = train_test_split(X_train_seq_trunc, y_train_oh, test_size=0.1, random_state=37)
    
        #print(X_train_emb)
        
        # Train the model
        emb_history = deep_model(model, X_train_emb, y_train_emb, X_valid_emb, y_valid_emb, NB_START_EPOCHS, BATCH_SIZE)
        
        # Get the test result
        emb_results = test_model(model, X_train_seq_trunc, y_train_oh, X_test_seq_trunc, y_test_oh, 5, BATCH_SIZE)
        print('/n')
        print('Test accuracy of word embeddings model: {0:.2f}%'.format(emb_results[1]*100))
        
        result.append(emb_results[1]*100)
        
    return result