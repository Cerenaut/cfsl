import torch

import numpy as np

def set_torch_seed(seed):
  """
  Sets the pytorch seeds for current experiment run
  :param seed: The seed (int)
  :return: A random number generator to use
  """
  rng = np.random.RandomState(seed=seed)
  torch_seed = rng.randint(0, 999999)
  torch.manual_seed(seed=torch_seed)

  return rng


def calculate_cosine_distance(support_set_embeddings, support_set_labels, target_set_embeddings):
  eps = 1e-10

  # Ensure both embeddings are on same device
  support_set_embeddings = support_set_embeddings.to(target_set_embeddings.device)

  per_task_similarities = []
  for support_set_embedding_task, target_set_embedding_task in zip(support_set_embeddings, target_set_embeddings):
    target_set_embedding_task = target_set_embedding_task  # sb, f
    support_set_embedding_task = support_set_embedding_task  # num_classes, f

    dot_product = torch.stack(
        [torch.matmul(target_set_embedding_task, support_vector) for support_vector in support_set_embedding_task],
        dim=1)
    cosine_similarity = dot_product

    cosine_similarity = cosine_similarity.squeeze()
    per_task_similarities.append(cosine_similarity)

  similarities = torch.stack(per_task_similarities)
  preds = similarities

  return preds, similarities
