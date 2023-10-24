import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Работа с пропущенными значениями
titanic_original = pd.read_csv('/Users/elizaveta/Downloads/titanic.csv')
pclass_na = titanic_original['Pclass'].isnull().sum()
age_na = titanic_original['Age'].isnull().sum()  # есть пустые значения
fare_na = titanic_original['Fare'].isnull().sum()
subsp_na = titanic_original['SibSp'].isnull().sum()
sex_na = titanic_original['Sex'].isnull().sum()
parch_na = titanic_original['Parch'].isnull().sum()

# Заполним пустые значения Age средним возрастом
mean_age = (titanic_original['Age'].mean()).round()
titanic_original['Age'].fillna(value=mean_age, inplace=True)
age_na = titanic_original['Age'].isnull().sum()

# One hot encoding для значений пола
dummies = pd.get_dummies(titanic_original['Sex'], prefix='Sex', dtype=int)
titanic_new = pd.concat([titanic_original, dummies], axis=1)
titanic_new.drop('Sex', axis=1, inplace=True)
print(titanic_new)


torch.manual_seed(2023)

class TitanicDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.df = titanic_new


    def __len__(self):
        return self.df.shape[0]



    def __getitem__(self, item):
        row = self.df.iloc[item]
        alive = torch.Tensor([1, 0])
        dead = torch.Tensor([0, 1])
        y = alive if row['Survived'] else dead
        x = torch.Tensor([row['Age'], row['Fare'], row['SibSp'], row['Pclass'],
                          row['Sex_female'], row['Sex_male'], row['Parch']])
        return x, y


titanic_dataset = TitanicDataset()
dataloader = DataLoader(dataset=titanic_dataset, batch_size=800, shuffle=True)

class SurvivalPredictorPerceptron(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.fully_connected_layer = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.out_layer = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()


    def forward(self, x):
        x = self.fully_connected_layer(x)
        x = self.relu(x)
        x = self.out_layer(x)
        x = self.sigmoid(x)
        x = self.softmax(x)
        return x

model = SurvivalPredictorPerceptron(input_size=7, hidden_size=50, output_size=2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 50

for epoch in range(num_epochs):
    error = 0
    for x,y in dataloader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        error += loss

        loss.backward()
        optimizer.step()
    print('Mean error', error/len(titanic_dataset))


# Сильнее всего уменьшало ошибку изменение параметра batch_size и learning rate