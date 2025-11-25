# app/modules/topic_segmentation.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re


def segment_topics(text: str, speaker_segments=None, num_topics: int = 3):
    """
    Simple unsupervised topic segmentation using TF-IDF + KMeans.
    Returns dict with topics and their representative sentences.

    For short transcripts or very few sentences, we gracefully fall back
    to a single-topic summary instead of forcing bad clusters.
    """
    print("Segmenting topics...")

    text = (text or "").strip()
    if not text:
        print("✗ Topic segmentation skipped (empty transcript)")
        return {
            "topics": [],
            "num_topics": 0,
        }

    # ------------------------------------------------------------------
    # 1. Handle very short transcripts as a single topic
    # ------------------------------------------------------------------
    word_count = len(text.split())
    if word_count < 40:
        print("⚠ Transcript too short for reliable topic segmentation – using a single topic")
        summary = text[:500]
        return {
            "topics": [
                {
                    "topic_id": 0,
                    "summary": summary,
                    "sentence_count": 1,
                }
            ],
            "num_topics": 1,
        }

    try:
        # ------------------------------------------------------------------
        # 2. Split into sentences
        # ------------------------------------------------------------------
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) < 3:
            print("⚠ Not enough sentences for topic segmentation – using a single topic")
            joined = " ".join(sentences) if sentences else text
            summary = joined[:500]
            return {
                "topics": [
                    {
                        "topic_id": 0,
                        "summary": summary,
                        "sentence_count": len(sentences) if sentences else 1,
                    }
                ],
                "num_topics": 1,
            }

        # If we don't have enough sentences to form multiple topics,
        # just treat everything as one.
        if len(sentences) <= num_topics:
            print("⚠ Too few distinct sentences to form multiple topics – using a single topic")
            joined = " ".join(sentences)
            summary = joined[:500]
            return {
                "topics": [
                    {
                        "topic_id": 0,
                        "summary": summary,
                        "sentence_count": len(sentences),
                    }
                ],
                "num_topics": 1,
            }

        # ------------------------------------------------------------------
        # 3. Vectorize sentences with TF-IDF
        # ------------------------------------------------------------------
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=200,
            min_df=1,
            ngram_range=(1, 2),  # include some bigrams for better topic separation
        )

        X = vectorizer.fit_transform(sentences)

        # ------------------------------------------------------------------
        # 4. Cluster with KMeans
        # ------------------------------------------------------------------
        # Avoid asking for more clusters than we can support
        # e.g. with 10 sentences and num_topics=3, n_clusters=3 is fine,
        # but with 5 sentences, 3 clusters is already quite aggressive.
        n_clusters = min(num_topics, max(1, len(sentences) // 2))
        if n_clusters <= 1:
            print("⚠ Data not rich enough for clustering – using a single topic")
            joined = " ".join(sentences)
            summary = joined[:500]
            return {
                "topics": [
                    {
                        "topic_id": 0,
                        "summary": summary,
                        "sentence_count": len(sentences),
                    }
                ],
                "num_topics": 1,
            }

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X)

        # ------------------------------------------------------------------
        # 5. Extract representative sentences per cluster
        # ------------------------------------------------------------------
        topics = []
        for i in range(n_clusters):
            cluster_indices = np.where(model.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue

            cluster_sentences = [sentences[j] for j in cluster_indices]

            # Rank sentences in this cluster by similarity to the centroid
            centroid = model.cluster_centers_[i]
            cluster_vecs = X[cluster_indices]
            # cluster_vecs is sparse; dot with centroid gives similarity
            sims = cluster_vecs.dot(centroid)
            if hasattr(sims, "A1"):  # sparse -> numpy array
                sims = sims.A1

            ranked_indices = cluster_indices[np.argsort(-sims)]

            # Take top 3 representative sentences
            top_sents = [sentences[j] for j in ranked_indices[:3]]
            topic_text = ". ".join(top_sents)
            if len(topic_text) > 500:
                topic_text = topic_text[:500] + "..."

            topics.append(
                {
                    "topic_id": i,
                    "summary": topic_text,
                    "sentence_count": len(cluster_indices),
                }
            )

        # Sort topics by how many sentences they cover (most important first)
        topics.sort(key=lambda t: t["sentence_count"], reverse=True)

        print(f"✓ Segmented into {len(topics)} topics")
        return {
            "topics": topics,
            "num_topics": len(topics),
        }

    except Exception as e:
        print(f"✗ Topic segmentation error: {e}")
        # Fallback: single topic with truncated transcript
        return {
            "topics": [
                {
                    "topic_id": 0,
                    "summary": text[:500],
                    "sentence_count": 1,
                }
            ],
            "num_topics": 1,
        }
