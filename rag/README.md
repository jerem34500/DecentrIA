RAG — pipeline de catégorisation (v1) · Animals

Ce dossier présente la chaîne de classification multi-passes appliquée à une catégorie (ici : Animals).
Objectif : structurer un gros corpus en sous-catégories fiables, avant la phase RAG (découpage → embeddings → vecteur → index/retrieval).

Ce qui est déjà inclus (v1)

1) Pass1 — DeBERTa (zero-shot) : catégories
fichier : classification des fichiers en catégories ( Deberta -- zero-shot).py
rôle : classe ~210k fichiers bruts en catégories (et/ou sous-catégories) avec DeBERTa v3 zero-shot.
sortie : subcats_pass1/<Cat>/<Subcat>/ + needs_review/ + rapport CSV.
pourquoi DeBERTa ? bon compromis perf / zéro entraînement pour un 1er tri massif.

2) Pass1b — DeBERTa (zero-shot) : sous-catégorisation d’une catégorie
fichier : Sous-catégorisation d’une catégorie (DeBERTa -- zero-shot).py
rôle : prend tous les .txt d’une catégorie (ex. Animals) et les répartit en sous-catégories (ex. birds, fish, insects_arachnids, …).
décision : seuil de score + marge top-2 ; copie/déplacement configurable ; rapport CSV.

3) Pass2 — Hybride “règles + score” (reclassement)
fichier : Reclassement Pass2 -- Hybride "règle+score".py
rôle : ne parcourt que Animals/needs_review et tente un rattrapage via taxonomie latine + lexiques + heuristiques marines, etc.
mode : dry-run par défaut ; peut déplacer si validé.

4) Pass3 — Audit & Apply (needs-only)
fichier : Reclassement pass3 -- Audit & appli (needs_preview).py
rôle : assainissement final de needs_review avec :
exceptions non-bio (culture/sport/lieu/organisation/tech) → restent en needs_review s’il n’y a pas de taxonomie ;
preuve taxonomique obligatoire pour reclasser (Insecta, Arachnida, Aves, Mammalia, Actinopterygii, …) ;
mots courts (ant/bee/wasp/...) pris seulement en mot entier + contexte taxonomique (±80 caractères).
sorties : rapport d’audit + manifest CSV ; avec --apply, déplace réellement.

Référentiel lexical
fichier : Liste des mots clé pour catégorie et sous catégories
rôle : liste de mots-clés/expressions par (sous)catégorie, utilisée par Pass2/Pass3 pour le scoring et les rapports (sans pondération figée).

Ce que nous allons ajouter (v2 — RAG complet)

Découpage & normalisation (chunking)
Nettoyage HTML léger, normalisation unicode, découpe en chunks cohérents (titres, paragraphes, fenêtres glissantes).

Embeddings
Génération d’embeddings (modèles ouverts adaptés au FR/EN, p.ex. bge-small / e5-base).
Paramétrage taille de chunk, overlap, pooling.

Vectorisation & Index
Stockage dans une base vecteur (ChromaDB/FAISS), méta-données (catégorie, source, chemin).
Construction du retriever (top-k, filtre par catégorie).

RAG exécution
Chaîne “retrieval → contexte → LLM” + garde-fous (limite tokens, citations).
Scripts d’inférence locaux + réglages (température, max tokens).

Évaluation & Observabilité
Jeux d’échantillons anonymisés, vérifications de précision, couverture, taux de refus ; métriques simples + tableaux.

Industrialisation légère
Fichiers config (chemins, seuils, modèles), CLI homogène, CI rapide (lint + dry-run), README de déploiement.
