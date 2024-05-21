#!/usr/bin/env wolframscript
(* ::Package:: *)

dataPath = FileNameJoin[{NotebookDirectory[], "lakers.csv"}];
data = Import[dataPath, "CSV"];


data


(* \:63d0\:53d6\:6570\:503c\:7279\:5f81\:ff0c\:5ffd\:7565\:7403\:5458\:59d3\:540d *)
features = Rest[data];

(* \:4f7f\:7528 K-means \:7b97\:6cd5\:8fdb\:884c\:805a\:7c7b\:ff0c\:8bbe\:7f6e\:805a\:7c7b\:6570\:91cf\:4e3a 3 *)
clusters = FindClusters[features, 3];

(* \:6253\:5370\:6bcf\:4e2a\:805a\:7c7b\:7684\:7403\:5458\:5217\:8868 *)
Print["Cluster 1: ", clusters[[1]]];
Print["Cluster 2: ", clusters[[2]]];
Print["Cluster 3: ", clusters[[3]]];



