# Module RAG (Retrieval-Augmented Generation)

Ce module implémente le RAG structuré de DecentrIA : 
il permet de connecter le LLM local à une base documentaire externe, afin d'enrichir les réponses avec des données fiables et actualisées.

## contenu prévu

- Extraction et nettoyage des données (Wikipedia, Gutenberg, Corpus légaux..)
- Catégorisation automatique (multi-passes strict/intermédiaire/relâché)
- Découpage en chunks
- Génération des embeddings
- Vectorisation et indexation (via ChromaDB ou équivalent)

### Objectif 

Donner au modèle une mémoire augmentée et structurée pour fournir des réponses précises, vérifiables et contextualisées.
