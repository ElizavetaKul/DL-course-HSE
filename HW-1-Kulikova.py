# задача: необходимо предсказать вероятность поступления абитуриента на основе
# результатов двух экзаменов и среднего балла

import torch

applicant_1 = torch.tensor([[7.0, 7.0, 6.0]])
applicant_2 = torch.tensor([[9.0, 8.0, 8.0]])
applicant_3 = torch.tensor([[4.0, 5.0, 3.0]])
applicant_4 = torch.tensor([[2.0, 5.0, 6.0]])
applicant_5 = torch.tensor([[10.0, 9.0, 8.0]])
applicant_6 = torch.tensor([[3.0, 4.0, 6.0]])

dataset = [
    (applicant_1, torch.tensor([[0.7]])),
    (applicant_2, torch.tensor([[0.8]])),
    (applicant_3, torch.tensor([[0.3]])),
    (applicant_4, torch.tensor([[0.4]])),
    (applicant_5, torch.tensor([[0.9]])),
    (applicant_6, torch.tensor([[0.2]]))
]

torch.manual_seed(1234)

weights = torch.rand((1, 3), requires_grad=True)   # параметров теперь 3, значит весов тоже нужно 3
bias = torch.rand((1, 1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=1e-3)

def predict_acceptance(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias

def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)

num_epochs = 12

for i in range(num_epochs):
    for x, y in dataset:
        optimizer.zero_grad()
        acceptance = predict_acceptance(x)
        loss = calc_loss(acceptance, y)
        loss.backward()
        print(loss)
        optimizer.step()
