# DecentrIA

 DecentrIA est un assistant IA hors ligne et open-source combinant :

 - LLM local (Large Language Model exécuté en local, sans cloud)
 - RAG structuré (catégorisation et recherche augmentée)
 - Interface graphique moderne (sobre et fonctionnelle)
 - Intégration web3 (connexion sécurisée via AES-256 et wallet blockchain)

L'objectif est d'offrir un assistant souverain,résilient et décentralisé, qui fonctionne même sans connexion Internet et reste conforme aux normes européennes (RGPD, MiCA)

## Fonctionnalités principales

-LLM local :
     - Support des modèles GGUF (Mistral, DeepSeek, LlaMA.cpp...)  
     - Exécution GPU optimisée (CUDA, ROCm)
     - 100% hors ligne
    
- RAG structuré
     - Méthodologie de tri et catégorisation :
         -Premier passage strict : Classification avec seuils élevés (catégorie et sous catégorie)
                        Objectif : Ne conserver que les fichiers dont la catégorisation est certaine.
                        Résultat : Fichiers bien classés >> ASSIGNED, les autres >> NEEDS_REVIEW

         -Second passage intermédiaire : Relance de la classification uniquement sur les fichiers en NEEDS_REVIEW
                                         - Seuils assouplis pour élargir la reconnaissance des catégories
                                         - Résultat : une partie des fichiers reclassés, les autres restent en REVIEW

       - Dernier passage souple : Dernière tentative avec seuil minimal
                                  - vise à réduire au maximum le nombres de fichiers non classés
                                  - ce qui reste est regroupé dans une catégorie "Autres" pour assurer une exhaustivité.

     - Techniquement :
         - Utilisation de modèles NLI (deberta-v3-large-mnli) capable de lire et comprendre le contenu
         - Découpage des fichiers en chunks (512 tokens avec chevauchement)
         - Évaluation par similarité sémantique >> choix de la meilleure catégorie et sous catégorie
         - Production d'un rapport à chaque passe : nombre de fichiers "ASSIGNED", "NEEDS_REVIEW", "UNASSIGNED".

- Web3 Wallet
     - Connexion sécurisée via AES-256
     - Fonctionnalités prévues :
         - Signature locale des transactions
         - compatibilité avec Metamask/Rabby
         - Intégration directe avec les smart contracts

### architecture

DecentrIA/
   >> llm_local/
      >> Module LLM (offline,GPU,quantisation)
   >> RAG/
      >> Module RAG (tri,chunking,embeddings,indexation]
   >> gui/
      >> Interface graphique (sobre et fonctionnelle)
   >> web3_wallet/
      >> sécurité AES-256 (fonctionnalités prévues)
   >> LICENCE
      >> Licence Apache 2.0
   >> README.md
      >> Documentation principale
   > .gitignore

#### Licence

Ce projet est distribué sous licence Apache 2.0, usage libre, modification et redistribution autorisés, sous réserve de conserver la licence et la mention des auteurs

##### Contributeurs

Jerem34500 : Fondateur & développeur principal
ChatGPT (OpenAI) : Support IA, architecture et co-rédaction technique

###### Vision

DecentrIA vise à demontrer qu'un autodidacte + une IA peuvent concevoir un projet robuste, professionnel et open-source, à la croisée de 
     - l'intelligence artificielle
     - la blockchain
     - l'autonomie numérique européenne
       
