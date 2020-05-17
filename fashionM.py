#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def plot_img(x, y, index):
    sample = x.iloc[index].values   
    sample = sample.reshape(28, 28) # 이미지 출력 위해 모양 변환
    plt.imshow(sample)
    plt.axis("off") # x, y축 없애기
    plt.show() # 이미치 출력
    print("Correct Label : ", y.iloc[index]) # 위의 이미지에 해당하는 정답 레이블 출력
   
##############################################################################################   
# 데이터 읽기
train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

# 데이터 샘플 수, 컬럼 수(정답 레이블, 28 * 28 픽셀) 확인
print(train.info())
print(test.info())

# train, test가 각 레이블 별로 고르게 분포
print(train["label"].value_counts())
print(test["label"].value_counts())

# head 함수로 데이터 일부분 출력
# 레이블과 픽셀1 ~ 픽셀 784 컬럼 존재
print(train.head())
print(test.head())

# train에서 데이터와 정답을 분리
train_x = train.drop("label", axis=1)
train_y = train["label"].copy()

# 분리 데이터 일부분 출력
# x에는 정답 레이블을 제외한 픽셀 정보, y에는 정답 레이블만 존재
print(train_x.head())
print(train_y.head())

# test에서 데이터와 정답을 분리
# 위와 마찬가지
test_x = test.drop("label", axis=1)
test_y = test["label"].copy()

print(test_x.head())
print(test_y.head())

##############################################################################################

# 이미지 샘플 하나 출력(9번 앵클부츠)
sample = train_x.iloc[35000].values
print(sample) # 넘파이 배열로 출력(픽셀값 0 ~ 255 확인)
sample = sample.reshape(28, 28) # 이미지 출력 위해 모양 변환
plt.imshow(sample)
plt.axis("off") # x, y축 없애기
plt.show() # 이미치 출력
print(train_y.iloc[35000]) # 위의 이미지에 해당하는 정답 레이블 출력

##############################################################################################

# 스케일링 1 : 픽셀 값이 0~255이므로 단순히 255로 나눈다.
# 정답 레이블은 스케일링 불필요
train_x_sca1 = train_x / 255.0
test_x_sca1 = test_x / 255.0

# 스케일링 2 : StandardScaler() 사용
std = StandardScaler()
train_x_sca2 = std.fit_transform(train_x)
test_x_sca2 = std.fit_transform(test_x)

print("="*20, "스케일링 x", "="*20)
##############################################################################################
# 결정트리 깊이 10 에 따른 결과
# 깊이 2는 정확도가 35% 정도, 깊이 20은 정확도 +-0.5% 차이
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=45)
scores = cross_val_score(tree_clf, train_x, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=45)
tree_clf.fit(train_x, train_y)
pred_y = tree_clf.predict(test_x)
print(accuracy_score(pred_y, test_y))
##############################################################################################
# 로지스틱 회귀에서 파라미터 multinomial 주고 소프트맥스 회귀 모델 사용
# 스케일 x : 85.39, 스케일 1 : 85.63, 스케일 2 : 85.13
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, n_jobs=-1, random_state=45)
scores = cross_val_score(softmax_reg, train_x, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, n_jobs=-1, random_state=45)
softmax_reg.fit(train_x, train_y)
pred_y = softmax_reg.predict(test_x)
print(accuracy_score(pred_y, test_y))

##############################################################################################
# 랜덤 포레스트 분류기
# 스케일 x : 85.97, 스케일 1 : 85.96, 스케일 2 : 85.81
forest_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
scores = cross_val_score(softmax_reg, train_x, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

forest_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
forest_clf.fit(train_x, train_y)
pred_y = forest_clf.predict(test_x)
print(accuracy_score(pred_y, test_y))

##############################################################################################
bag_clf = BaggingClassifier(tree_clf, n_estimators=10, n_jobs=-1, random_state=45, bootstrap=True,
                            oob_score=True)

bag_clf.fit(train_x, train_y)
print(bag_clf.oob_score_)

pred_y = bag_clf.predict(test_x)
print(accuracy_score(pred_y, test_y))

##############################################################################################
voting_clf = VotingClassifier(estimators=[('softmax_reg', softmax_reg), ('forest_clf', forest_clf),
                                          ('bag_clf', bag_clf)], voting='soft', n_jobs=-1)
scores = cross_val_score(voting_clf, train_x, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())
voting_clf.fit(train_x, train_y)
for clf in (softmax_reg, forest_clf, bag_clf, voting_clf):
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    print(clf.__class__.__name__, accuracy_score(pred_y, test_y))
##############################################################################################
print("="*48)



print("="*20, "스케일링 1", "="*20)
##############################################################################################
# 결정트리 깊이 10 에 따른 결과
# 깊이 2는 정확도가 35% 정도, 깊이 20은 정확도 +-0.5% 차이
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=45)
scores = cross_val_score(tree_clf, train_x_sca1, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=45)
tree_clf.fit(train_x_sca1, train_y)
pred_y = tree_clf.predict(test_x_sca1)
print(accuracy_score(pred_y, test_y))
##############################################################################################
# 로지스틱 회귀에서 파라미터 multinomial 주고 소프트맥스 회귀 모델 사용
# 스케일 x : 85.39, 스케일 1 : 85.63, 스케일 2 : 85.13
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, n_jobs=-1, random_state=45)
scores = cross_val_score(softmax_reg, train_x_sca1, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, n_jobs=-1, random_state=45)
softmax_reg.fit(train_x_sca1, train_y)
pred_y = softmax_reg.predict(test_x_sca1)
print(accuracy_score(pred_y, test_y))

##############################################################################################
# 랜덤 포레스트 분류기
# 스케일 x : 85.97, 스케일 1 : 85.96, 스케일 2 : 85.81
forest_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
scores = cross_val_score(softmax_reg, train_x_sca1, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

forest_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
forest_clf.fit(train_x_sca1, train_y)
pred_y = forest_clf.predict(test_x_sca1)
print(accuracy_score(pred_y, test_y))

##############################################################################################
bag_clf = BaggingClassifier(tree_clf, n_estimators=10, n_jobs=-1, random_state=45, bootstrap=True,
                            oob_score=True)

bag_clf.fit(train_x_sca1, train_y)
print(bag_clf.oob_score_)

pred_y = bag_clf.predict(test_x_sca1)
print(accuracy_score(pred_y, test_y))

##############################################################################################
voting_clf = VotingClassifier(estimators=[('softmax_reg', softmax_reg), ('forest_clf', forest_clf),
                                          ('bag_clf', bag_clf)], voting='soft', n_jobs=-1)
scores = cross_val_score(voting_clf, train_x_sca1, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())
voting_clf.fit(train_x_sca1, train_y)
for clf in (softmax_reg, forest_clf, bag_clf, voting_clf):
    clf.fit(train_x_sca1, train_y)
    pred_y = clf.predict(test_x_sca1)
    print(clf.__class__.__name__, accuracy_score(pred_y, test_y))
##############################################################################################
print("="*48)



print("="*20, "스케일링 2", "="*20)
##############################################################################################
# 결정트리 깊이 10 에 따른 결과
# 깊이 2는 정확도가 35% 정도, 깊이 20은 정확도 +-0.5% 차이
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=45)
scores = cross_val_score(tree_clf, train_x_sca2, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=45)
tree_clf.fit(train_x_sca2, train_y)
pred_y = tree_clf.predict(test_x_sca2)
print(accuracy_score(pred_y, test_y))
##############################################################################################
# 로지스틱 회귀에서 파라미터 multinomial 주고 소프트맥스 회귀 모델 사용
# 스케일 x : 85.39, 스케일 1 : 85.63, 스케일 2 : 85.13
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, n_jobs=-1, random_state=45)
scores = cross_val_score(softmax_reg, train_x_sca2, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, n_jobs=-1, random_state=45)
softmax_reg.fit(train_x_sca2, train_y)
pred_y = softmax_reg.predict(test_x_sca2)
print(accuracy_score(pred_y, test_y))

##############################################################################################
# 랜덤 포레스트 분류기
# 스케일 x : 85.97, 스케일 1 : 85.96, 스케일 2 : 85.81
forest_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
scores = cross_val_score(softmax_reg, train_x_sca2, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

forest_clf = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
forest_clf.fit(train_x_sca2, train_y)
pred_y = forest_clf.predict(test_x_sca2)
print(accuracy_score(pred_y, test_y))

##############################################################################################
bag_clf = BaggingClassifier(tree_clf, n_estimators=10, n_jobs=-1, random_state=45, bootstrap=True,
                            oob_score=True)

bag_clf.fit(train_x_sca2, train_y)
print(bag_clf.oob_score_)

pred_y = bag_clf.predict(test_x_sca2)
print(accuracy_score(pred_y, test_y))

##############################################################################################
voting_clf = VotingClassifier(estimators=[('softmax_reg', softmax_reg), ('forest_clf', forest_clf),
                                          ('bag_clf', bag_clf)], voting='soft', n_jobs=-1)
scores = cross_val_score(voting_clf, train_x_sca2, train_y, scoring="accuracy", cv=10)
print(scores)
print(scores.mean())

voting_clf.fit(train_x_sca2, train_y)
for clf in (softmax_reg, forest_clf, bag_clf, voting_clf):
    clf.fit(train_x_sca2, train_y)
    pred_y = clf.predict(test_x_sca2)
    print(clf.__class__.__name__, accuracy_score(pred_y, test_y))
##############################################################################################
print("="*48)





import tensorflow as tf
import pandas as pd
import numpy as np

def reset_graph(seed=42):
    # 텐서플로우의 그래프를 초기화하고 시드값을 설정
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
dropout = 0.5
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

he_init = tf.ariance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, kernel_initializer=he_init, name="hidden1")
hidden1_drop = tf.layers.dropout(hidden1, dropout, training=training)
hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", activation=tf.nn.elu)
hidden2_drop = tf.layers.dropout(hidden2, dropout, training=training)
logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

   

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")   
learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)


correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
   
init = tf.global_variables_initializer()
saver = tf.train.Saver()


train = pd.read_csv("fashion_train.csv")
test = pd.read_csv("fashion_test.csv")

X_train = train.drop("label", axis=1).values
y_train = train["label"].values.copy()
X_test = test.drop("label", axis=1).values
y_test = test["label"].values.copy()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config=tf.ConfigProto(gpu_options=gpu_options)

X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 50
batch_size = 100   



with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch Accuracy:", acc_batch, "Validation Accuracy:", acc_valid)   
       

    save_path = saver.save(sess, "./my_model_final.ckpt")

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = X_test[:10000] # 스케일 조정 안되어있다면, 해줘야함
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)

print("Predict :", y_pred)
print("Target  :", y_test[:10000])
print("Accuracy :", (y_pred == y_test[:10000]).sum() / len(y_pred))

