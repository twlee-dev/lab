#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
print('슝=3')


# In[2]:


import os
csv_path = os.getenv("HOME") +"/aiffel/pokemon_eda/data/Pokemon.csv"
original_data = pd.read_csv(csv_path)
print('슝=3')


# In[3]:


pokemon = original_data.copy()
print(pokemon.shape)
pokemon.head()


# In[4]:


# 전설의 포켓몬 데이터셋
legendary = pokemon[pokemon["Legendary"] == True].reset_index(drop=True)
print(legendary.shape)
legendary.head()


# In[5]:


# Q. 일반 포켓몬의 데이터셋도 만들어봅시다.
ordinary = pokemon[pokemon["Legendary"] == False].reset_index(drop=False)
print(ordinary.shape)
ordinary.head()


# In[6]:


pokemon.isnull().sum()


# In[7]:


print(len(pokemon.columns))
pokemon.columns


# In[8]:


len(set(pokemon["#"]))


# In[9]:


pokemon[pokemon["#"] == 6]


# In[10]:


# Q. 총 몇 종류의 포켓몬 이름이 있는지 확인해봅시다!
unique_names_count = len(set(pokemon["Name"]))

print(unique_names_count)


# In[11]:


pokemon.loc[[6, 10]]


# In[12]:


len(list(set(pokemon["Type 1"]))), len(list(set(pokemon["Type 2"])))


# In[13]:


set(pokemon["Type 2"]) - set(pokemon["Type 1"])


# In[14]:


types = list(set(pokemon["Type 1"]))
print(len(types))
print(types)


# In[15]:


pokemon["Type 2"].isna().sum()


# In[16]:


plt.figure(figsize=(10, 7))  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=ordinary, x="Type 1", order=types).set_xlabel('')
plt.title("[Ordinary Pokemons]")

plt.subplot(212)
sns.countplot(data=legendary, x="Type 1", order=types).set_xlabel('')
plt.title("[Legendary Pokemons]")

plt.show()


# In[17]:


# Type1별로 Legendary의 비율을 보여주는 피벗 테이블
pd.pivot_table(pokemon, index="Type 1", values="Legendary").sort_values(by=["Legendary"], ascending=False)


# In[18]:


# Q. 아래 코드의 빈칸을 채워주세요.
plt.figure(figsize=(12, 10))  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=ordinary, x="Type 2", order=types).set_xlabel('')
plt.title("[Ordinary Pokemons]")

plt.subplot(212)
sns.countplot(data=legendary, x="Type 2", order=types).set_xlabel('')
plt.title("[Legendary Pokemons]")

plt.show()


# In[19]:


# Q. Type 2에 대해서도 피벗 테이블을 만들어봅시다.
pd.pivot_table(pokemon, index="Type 2", values="Legendary")   .sort_values(by=["Legendary"], ascending=False)


# In[20]:


stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
stats


# In[21]:


print("#0 pokemon: ", pokemon.loc[0, "Name"])
print("total: ", int(pokemon.loc[0, "Total"]))
print("stats: ", list(pokemon.loc[0, stats]))
print("sum of all stats: ", sum(list(pokemon.loc[0, stats])))


# In[22]:


# 능력치 컬럼 리스트
stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# Total 값과 stats 합계가 같은 포켓몬 수 확인
same_count = sum(pokemon["Total"].values == pokemon[stats].sum(axis=1).values)

print(same_count)


# In[23]:


fig, ax = plt.subplots()
fig.set_size_inches(12, 6)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

sns.scatterplot(data=pokemon, x="Type 1", y="Total", hue="Legendary")
plt.show()


# In[24]:


figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(12, 18)  # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

# "HP" 스탯의 scatter plot
sns.scatterplot(data=pokemon, y="Total", x="HP", hue="Legendary", ax=ax1)

# "Attack" 스탯의 scatter plot
sns.scatterplot(data=pokemon, y="Total", x="Attack", hue="Legendary", ax=ax2)

# "Defense" 스탯의 scatter plot
sns.scatterplot(data=pokemon, y="Total", x="Defense", hue="Legendary", ax=ax3)

# "Sp. Atk" 스탯의 scatter plot
sns.scatterplot(data=pokemon, y="Total", x="Sp. Atk", hue="Legendary", ax=ax4)

# "Sp. Def" 스탯의 scatter plot
sns.scatterplot(data=pokemon, y="Total", x="Sp. Def", hue="Legendary", ax=ax5)

# "Speed" 스탯의 scatter plot
sns.scatterplot(data=pokemon, y="Total", x="Speed", hue="Legendary", ax=ax6)

plt.show()


# In[25]:


plt.figure(figsize=(12, 10))   # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=ordinary, x="Generation").set_xlabel('')
plt.title("[Ordinary Pokemons]")
plt.subplot(212)
sns.countplot(data=legendary, x="Generation").set_xlabel('')
plt.title("[Legendary Pokemons]")
plt.show()


# In[26]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

sns.scatterplot(data=legendary, y="Type 1", x="Total")
plt.show()


# In[27]:


print(sorted(list(set(legendary["Total"]))))


# In[28]:


fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

sns.countplot(data=legendary, x="Total")
plt.show()


# In[29]:


round(65 / 9, 2)


# In[30]:


# Q. ordinary 포켓몬의 'Total' 값 집합을 확인해봅시다.
print(sorted(list(set(ordinary["Total"]))))


# In[31]:


# Q. 이 집합의 크기(길이)를 확인해봅시다.
print(len(set(ordinary["Total"])))


# In[32]:


round(735 / 195, 2)


# In[33]:


n1, n2, n3, n4, n5 = legendary[3:6], legendary[14:24], legendary[25:29], legendary[46:50], legendary[52:57]
names = pd.concat([n1, n2, n3, n4, n5]).reset_index(drop=True)
names


# In[34]:


formes = names[13:23]
formes


# In[35]:


legendary["name_count"] = legendary["Name"].apply(lambda i: len(i))    
legendary.head()


# In[36]:


# Q. ordinary 포켓몬의 데이터에도 'name_count' 값을 추가해줍시다.
ordinary["name_count"] = ordinary["Name"].apply(lambda i: len(i))    
ordinary.head()


# In[37]:


plt.figure(figsize=(12, 10))   # 화면 해상도에 따라 그래프 크기를 조정해 주세요.

plt.subplot(211)
sns.countplot(data=legendary, x="name_count").set_xlabel('')
plt.title("Legendary")
plt.subplot(212)
sns.countplot(data=ordinary, x="name_count").set_xlabel('')
plt.title("Ordinary")
plt.show()


# In[38]:


print(round(len(legendary[legendary["name_count"] > 9]) / len(legendary) * 100, 2), "%")


# In[39]:


pokemon["name_count"] = pokemon["Name"].apply(lambda i: len(i))
pokemon.head()


# In[40]:


pokemon["long_name"] = pokemon["name_count"] >= 10
pokemon.head()


# In[41]:


pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon.tail()


# In[42]:


pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
pokemon.head()


# In[43]:


print(pokemon[pokemon["name_isalpha"] == False].shape)
pokemon[pokemon["name_isalpha"] == False]


# In[44]:


pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="Porygon Two")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")

pokemon.loc[[34, 37, 90, 131, 252, 270, 487, 525, 794]]


# In[45]:


# Q. 바꿔준 'Name' 컬럼으로 'Name_nospace'를 만들고, 다시 isalpha()로 체크해봅시다.
pokemon["Name_nospace"] = pokemon["Name"].str.replace(" ", "")
pokemon["name_isalpha"] = pokemon["Name_nospace"].str.isalpha()
pokemon[pokemon["name_isalpha"] == False]


# In[46]:


import re


# In[47]:


name = "CharizardMega Charizard X"


# In[48]:


name_split = name.split(" ")
name_split


# In[49]:


temp = name_split[0]
temp


# In[50]:


tokens = re.findall('[A-Z][a-z]*', temp)
tokens


# In[51]:


tokens = []
for part_name in name_split:
    a = re.findall('[A-Z][a-z]*', part_name)
    tokens.extend(a)
tokens


# In[53]:


def tokenize(name):
    tokens = []
    for part_name in name.split():
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a)
    return np.array(tokens)


# In[54]:


name = "CharizardMega Charizard X"
tokenize(name)


# In[55]:


all_tokens = list(legendary["Name"].apply(tokenize).values)

token_set = []
for token in all_tokens:
    token_set.extend(token)

print(len(set(token_set)))
print(token_set)


# In[56]:


from collections import Counter


# In[57]:


a = [1, 1, 0, 0, 0, 1, 1, 2, 3]
Counter(a)


# In[58]:


Counter(a).most_common()


# In[59]:


most_common = Counter(token_set).most_common(10)
most_common


# In[60]:


for token, _ in most_common:
    # pokemon[token] = ... 형식으로 사용하면 뒤에서 warning이 발생합니다
    pokemon[f"{token}"] = pokemon["Name"].str.contains(token)

pokemon.head(10)


# In[61]:


print(types)


# In[62]:


for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)
    
pokemon[[["Type 1", "Type 2"] + types][0]].head()


# In[63]:


print(original_data.shape)
original_data.head()


# In[64]:


original_data.columns


# In[65]:


features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']


# In[ ]:


target = 'Legendary'


# In[66]:


# Q. 'original_data'에서 'features' 컬럼에 해당하는 데이터를 변수 'X'에 저장합니다.
X = original_data[features]
print(X.shape)
X.head()


# In[72]:


print(original_data.shape)
original_data.head()


# In[73]:


original_data.columns


# In[74]:


features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']


# In[75]:


target = 'Legendary'


# In[76]:


# Q. 'target' 컬럼의 데이터를 변수 'y'에 저장합니다.
y = original_data[target].copy()
print(y.shape)
y.head()


# In[77]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[78]:


from sklearn.tree import DecisionTreeClassifier
print('슝=3')


# In[79]:


model = DecisionTreeClassifier(random_state=25)
model


# In[80]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('슝=3')


# In[81]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[84]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
len(legendary)


# In[85]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[86]:


print(len(pokemon.columns))
print(pokemon.columns)


# In[87]:


features = ['Total', 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 
            'name_count','long_name', 'Forme', 'Mega', 'Mewtwo','Deoxys', 'Kyurem', 'Latias', 'Latios',
            'Kyogre', 'Groudon', 'Hoopa','Poison', 'Ground', 'Flying', 'Normal', 'Water', 'Fire',
            'Electric','Rock', 'Dark', 'Fairy', 'Steel', 'Ghost', 'Psychic', 'Ice', 'Bug', 'Grass', 'Dragon', 'Fighting']

len(features)


# In[88]:


target = "Legendary"
target


# In[89]:


# Q. 사용할 feature에 해당하는 데이터를 'X' 변수에 저장합니다.
X = original_data[target]
print(X.shape)
X.head()


# In[90]:


# Q. 정답 데이터 'y'도 'target' 변수를 이용해 만들어줍시다.
Y = original_data[target]
print(y.shape)
y.head()


# In[91]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[92]:


model = DecisionTreeClassifier(random_state=25)
model


# In[105]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('슝=3')


# In[111]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[109]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
len(legendary)


# In[110]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




