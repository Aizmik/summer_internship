from utilities import generate_data
from seq2seq import Seq2seq


seq2seq = Seq2seq(lr=0.03, init_range=0.5)
epochs = 80
train_length = 1000
test_length = 100

# generating data
max_array_length = 2
input_train, output_train = generate_data(train_length, max_array_length)
input_test, output_test = generate_data(test_length, max_array_length)

# model training
for epoch in range(epochs):
    cost = 0
    for j in range(len(input_train)):
        cost += seq2seq.train(input_train[j], output_train[j])

    if epoch % 5 == 0:
        print(f'Epoch: {epoch}')
        print(f'loss: {cost}')
        print('-' * 25)

# model evaluation
correct_test = 0
print('\ntest results:')
for l in range(test_length):
    prediction = seq2seq.predict(input_test[l])
    print(input_test[l], end=' ')
    if prediction == output_test[l]:
        correct_test += 1
        print('☑', end=' ')
    else:
        print('☒', end=' ')
    print(prediction)
print('test accuracy', correct_test / test_length)
