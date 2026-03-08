"""
Evaluation utilities for dog image matching.
Includes similarity search, re-ranking, and accuracy metrics.
"""
import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity

# FAISS is optional - will fall back to cosine similarity if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def compute_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    return_paths: bool = False,
    dataset_name: str = "dataset"
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Compute embeddings for all images in dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader
        device: Device to run on
        return_paths: Whether to return image paths (for excluding self-matches)
        dataset_name: Name of dataset for progress display
        
    Returns:
        (embeddings, dog_ids) or (embeddings, dog_ids, paths) if return_paths=True
    """
    model.eval()
    embeddings = []
    dog_ids = []
    paths = [] if return_paths else None
    
    total_batches = len(dataloader)
    print(f'  Computing embeddings for {dataset_name} ({total_batches} batches)...')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f'  Progress: {batch_idx + 1}/{total_batches} batches processed', end='\r')
            
            if 'frontal' in batch and 'lateral' in batch:
                # Dual-view model
                frontal = batch['frontal'].to(device)
                lateral = batch['lateral'].to(device)
                emb = model(frontal, lateral)
            else:
                # Single-view model
                images = batch['image'].to(device)
                emb = model(images)
            
            embeddings.append(emb.cpu())
            dog_ids.extend(batch['dog_id'])
            
            if return_paths:
                # Get paths (use frontal_path if available, otherwise image path)
                if 'frontal_path' in batch:
                    paths.extend(batch['frontal_path'])
                elif 'path' in batch:
                    paths.extend(batch['path'])
                else:
                    paths.extend([''] * len(batch['dog_id']))
    
    print(f'  Completed: {total_batches}/{total_batches} batches processed')
    embeddings = torch.cat(embeddings, dim=0)
    if return_paths:
        return embeddings, dog_ids, paths
    else:
        return embeddings, dog_ids


def cosine_similarity_search(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    top_k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find top-k most similar embeddings using cosine similarity.
    
    Args:
        query_embedding: Query embedding [embedding_dim]
        gallery_embeddings: Gallery embeddings [N, embedding_dim]
        top_k: Number of top results to return
        
    Returns:
        (similarities, indices)
    """
    # Normalize
    query_norm = query_embedding / query_embedding.norm(p=2)
    gallery_norm = gallery_embeddings / gallery_embeddings.norm(p=2, dim=1, keepdim=True)
    
    # Compute cosine similarity
    similarities = torch.mm(query_norm.unsqueeze(0), gallery_norm.t()).squeeze(0)
    
    # Get top-k
    top_similarities, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
    
    return top_similarities, top_indices


def faiss_search(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    top_k: int = 10,
    use_gpu: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast similarity search using FAISS.
    
    Args:
        query_embeddings: Query embeddings [N_query, embedding_dim]
        gallery_embeddings: Gallery embeddings [N_gallery, embedding_dim]
        top_k: Number of top results
        use_gpu: Use GPU if available
        
    Returns:
        (distances, indices)
    """
    dimension = gallery_embeddings.shape[1]
    
    # Normalize for cosine similarity
    faiss.normalize_L2(gallery_embeddings)
    faiss.normalize_L2(query_embeddings)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # Add gallery embeddings
    index.add(gallery_embeddings.astype('float32'))
    
    # Search
    distances, indices = index.search(query_embeddings.astype('float32'), top_k)
    
    return distances, indices


def compute_accuracy_at_k(
    query_ids: List[str],
    gallery_ids: List[str],
    top_indices: np.ndarray,
    k_values: List[int] = [1, 5, 10],
    query_paths: List[str] = None,
    gallery_paths: List[str] = None
) -> Dict[int, float]:
    """
    Compute accuracy@k metrics.
    
    Args:
        query_ids: Query dog IDs
        gallery_ids: Gallery dog IDs
        top_indices: Top-k indices for each query [N_query, k]
        k_values: List of k values to compute
        query_paths: Query image paths (optional, for excluding self-matches)
        gallery_paths: Gallery image paths (optional, for excluding self-matches)
        
    Returns:
        Dictionary of {k: accuracy} and optionally {k: (correct, total)} counts
    """
    accuracies = {}
    accuracy_counts = {}  # Store (correct, total) for each k
    
    for k in k_values:
        if k > top_indices.shape[1]:
            continue
        
        correct = 0
        total = 0
        
        for i, query_id in enumerate(query_ids):
            # Get top-k matches
            top_k_indices = top_indices[i, :k]
            
            # Exclude self-matches if paths provided
            if query_paths and gallery_paths:
                query_path = os.path.normpath(query_paths[i])
                # Filter out self-matches (same image path, normalized for comparison)
                valid_indices = [idx for idx in top_k_indices 
                               if query_path != os.path.normpath(gallery_paths[idx])]
                
                if len(valid_indices) == 0:
                    # All matches were self-matches, count as incorrect
                    total += 1
                    continue
                
                # Check if correct dog is in remaining (non-self) matches
                valid_ids = [gallery_ids[idx] for idx in valid_indices]
                if query_id in valid_ids:
                    correct += 1
                total += 1
            else:
                # No path information, use original logic (may include self-matches)
                top_k_ids = [gallery_ids[idx] for idx in top_k_indices]
                if query_id in top_k_ids:
                    correct += 1
                total += 1
        
        accuracies[k] = correct / total if total > 0 else 0.0
        accuracy_counts[k] = (correct, total)
    
    return accuracies, accuracy_counts


def re_rank_results(
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    initial_scores: torch.Tensor,
    method: str = 'reciprocal'
) -> torch.Tensor:
    """
    Re-rank search results using reciprocal rank fusion or other methods.
    
    Args:
        query_embedding: Query embedding
        candidate_embeddings: Candidate embeddings
        initial_scores: Initial similarity scores
        method: Re-ranking method ('reciprocal', 'rerank')
        
    Returns:
        Re-ranked scores
    """
    if method == 'reciprocal':
        # Reciprocal rank fusion
        ranks = torch.argsort(initial_scores, descending=True)
        reciprocal_ranks = 1.0 / (torch.arange(len(ranks), dtype=torch.float32) + 1)
        reranked_scores = torch.zeros_like(initial_scores)
        reranked_scores[ranks] = reciprocal_ranks
        return reranked_scores
    
    elif method == 'rerank':
        # Re-compute with more detailed comparison
        # Could use additional features or different distance metric
        return initial_scores  # Placeholder
    
    else:
        return initial_scores


def evaluate_model(
    model: torch.nn.Module,
    query_loader: torch.utils.data.DataLoader,
    gallery_loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 10,
    use_faiss: bool = True
) -> Dict:
    """
    Evaluate model on query and gallery sets.
    
    Args:
        model: Trained model
        query_loader: Query dataloader
        gallery_loader: Gallery dataloader
        device: Device
        top_k: Top-k for retrieval
        use_faiss: Use FAISS for fast search
        
    Returns:
        Evaluation metrics dictionary
    """
    # Compute embeddings (with paths for self-match exclusion)
    print('Computing query embeddings...')
    query_embeddings, query_ids, query_paths = compute_embeddings(
        model, query_loader, device, return_paths=True, dataset_name="queries"
    )
    
    print('Computing gallery embeddings...')
    gallery_embeddings, gallery_ids, gallery_paths = compute_embeddings(
        model, gallery_loader, device, return_paths=True, dataset_name="gallery"
    )
    
    print(f'Query embeddings shape: {query_embeddings.shape}')
    print(f'Gallery embeddings shape: {gallery_embeddings.shape}')
    
    # Convert to numpy for FAISS
    print('Converting to numpy...')
    query_emb_np = query_embeddings.numpy()
    gallery_emb_np = gallery_embeddings.numpy()
    
    # Search
    print(f'Searching for top-{top_k} matches...')
    if use_faiss and FAISS_AVAILABLE:
        try:
            distances, indices = faiss_search(query_emb_np, gallery_emb_np, top_k=top_k)
            # Convert distances to similarities (FAISS returns inner product for normalized vectors)
            similarities = distances
            print('Search completed using FAISS')
        except Exception as e:
            print(f'FAISS search failed: {e}')
            print('Falling back to cosine similarity search...')
            use_faiss = False
    elif use_faiss and not FAISS_AVAILABLE:
        print('FAISS not available, using cosine similarity search...')
        use_faiss = False
    
    if not use_faiss:
        # Use cosine similarity
        print('Using cosine similarity search (this may take a while for large galleries)...')
        similarities_list = []
        indices_list = []
        for i in range(len(query_embeddings)):
            if (i + 1) % 10 == 0 or (i + 1) == len(query_embeddings):
                print(f'  Progress: {i + 1}/{len(query_embeddings)} queries processed', end='\r')
            sim, idx = cosine_similarity_search(query_embeddings[i], gallery_embeddings, top_k)
            similarities_list.append(sim.numpy())
            indices_list.append(idx.numpy())
        print(f'  Completed: {len(query_embeddings)}/{len(query_embeddings)} queries processed')
        similarities = np.array(similarities_list)
        indices = np.array(indices_list)
    
    # Compute accuracy metrics (FIX: exclude self-matches)
    print('Computing accuracy metrics...')
    accuracies, accuracy_counts = compute_accuracy_at_k(
        query_ids, gallery_ids, indices, 
        k_values=[1, 5, 10],
        query_paths=query_paths,
        gallery_paths=gallery_paths
    )
    
    result = {
        'accuracies': accuracies,
        'accuracy_counts': accuracy_counts,  # Add counts for detailed output
        'similarities': similarities,
        'indices': indices,
        'query_ids': query_ids,
        'gallery_ids': gallery_ids
    }
    if query_paths is not None:
        result['query_paths'] = query_paths
        result['gallery_paths'] = gallery_paths
    return result

