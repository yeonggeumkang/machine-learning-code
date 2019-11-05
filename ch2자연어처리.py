#2.3.1 파이썬으로 말뭉치 전처리하기
text = 'You say goodbye and I say hello.'
text = text.lower()
text = text.replace('.', ' .')
words = text.split(' ') #공백 기준으로 단어 분할

# word, id 딕셔너리 생성
word_to_id = {} #word 기준 ID
id_to_word = {} #ID 기준 word
for word in words :
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word

#ID로 이루어진 text
import numpy as np
corpus = [word_to_id[w] for w in words] #id값 뽑아오기
corpus = np.array(corpus)


#전처리함수 구현
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {} #word 기준 ID
    id_to_word = {} #ID 기준 word
    for word in words :
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = [word_to_id[w] for w in words]
    return corpus, word_to_id, id_to_word


#2.3.2 단어의 분산 표현 : 단어의 벡터화 ~ 2.3.4 동시발행 행렬

#동시발생 행렬 함수 구현
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    #size만큼의 0으로 이루어진 행렬 생성

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx-1
            right_idx = idx+1

            if left_idx>=0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] +=1

    return co_matrix

# 2.3.5 벡터 간 유사도
# 코사인 유사도 함수 구현
def cos_similarity(x,y, eps=1e-8): #x, y는 넘파이 배열
    nx = x/(np.sqrt(np.sum(x**2)) + eps) #x정규화
    ny = y/(np.sqrt(np.sum(y**2)) + eps) #y정규화
    return np.dot(nx, ny) #dot : 행렬 곱


# test
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
#print(cos_similarity(c0, c1))

#2.3.6 유사 단어의 랭킹 표시

# query : 기준 검색어, word_matrix : 단어 벡터 행렬 (C)
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):

    # 1 : 검색어 꺼내기
    if query not in word_to_id:
        print("%s(을)를 찾을 수 없습니다." %query)
        return
    print('\n[query] : ' + query) # 기준 단어(query)를 출력해 줌.
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    #2:코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size) #vocab_size만큼의 0 리스트 생성
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    #3: 내림차순 출력
    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s : %s' %(id_to_word[i], similarity[i]))
        count+=1
        if count >= top:
            return


#test
#most_similar('you', word_to_id, id_to_word, C, top=5)


#2.4 통계 기반 기법 개선하기 : 상호정보량
# PPMI(양의 상호 정보량) 함수 구현
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32) #C와 같은 형태의 0 행렬 생성
    N = np.sum(C) # C 원소값 총합
    S = np.sum(C, axis=0) #x축 기준 총합(??)
    total = C.shape[0] * C.shape[1] # C.shape (7,7), total 49
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[i]*S[j]) + eps)
            M[i,j] = max(0,pmi)


            if verbose:
                print(total)
                cnt +=1
                if cnt % (total//100) == 0: #ZeroDivisionError 발생
                    print('%.1f%% 완료' % (100*cnt/total))
    return M

#test
W = ppmi(C)


#2.4.2 차원감소 : SVD
U, S, V = np.linalg.svd(W)
#밀집벡터 U, 차원감소
print(U[0, :2])

#텍스트 그래프화
'''for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id,0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()'''
