###################################
###### For CNN-learning code ######
###################################


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib




if __name__=='__main__':
# 1. 데이터셋 생성하기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 데이터셋 구경하기
    '''   
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    cv2.imshow('mnist', x_train[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    print(len(x_train))
    #exit()
# 3. 데이터셋 전처리
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    #x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) # (60000, 28, 28) -> (60000, 28, 28, 1)
    #x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) # (10000, 28, 28) -> (10000, 28, 28, 1)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) # (60000, 28, 28) -> (60000, 28, 28, 1)
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) # (10000, 28, 28) -> (10000, 28, 28, 1)

    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# 4. validation나누기
    '''
    x_val = x_train[:10500]
    x_train = x_train[10500:]
    y_val = y_train[:10500]
    y_train = y_train[10500:]
    '''

    #5. 모델 생성하기
    model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy' , metrics=['accuracy'])
    
    model.summary()

    #6. 모델 학습,평가 및 저장
    epo = 1000

    hist = model.fit(x_train, y_train, epochs=epo, batch_size=32 ,validation_split=0.15)
    train_loss = hist.history['loss']
    train_acc = hist.history['accuracy']
    val_loss = hist.history['val_loss']
    val_acc = hist.history['val_accuracy']

    fig = plt.figure(figsize=(10,21)) # 도화지 생성

    ax_acc = fig.add_subplot(211) # 3x1, 1번째 서브플롯
    ax_acc.plot(range(epo), train_acc, label = 'train_acc(%)', color='darkred')
    mix_acc = ax_acc.twinx()
    mix_acc.plot(range(epo), val_acc, label = 'val_acc(%)', color='darkblue')
    ax_acc.grid(linestyle='--', color='lavender')
    ax_acc.legend()
    mix_acc.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    ax_loss = fig.add_subplot(212)
    ax_loss.plot(range(epo), train_loss, label = 'train_loss(%)', color='darkred')
    mix_loss = ax_loss.twinx()
    mix_loss.plot(range(epo), val_loss, label = 'val_loss(%)', color='darkblue') 
    ax_loss.grid(linestyle='--', color='lavender')
    ax_loss.legend()
    mix_loss.legend(loc='upper left')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    ax_acc.set_title('accuracy') 
    ax_loss.set_title('loss')
    fig.tight_layout()

    plt.show # 그래프 표시
    
    plt.savefig('CNN_acc.png') # 그래프 저장


    print('\nevaluate with test data:')

    loss_model = model.evaluate(x_test, y_test)

    print('\nevaluation : ' + str(loss_model))
    print('end')
    #model.save('CNN_2layers_model3.h5')
    model.save('CNN_2layers_model5.h5')
    
    print('end')
