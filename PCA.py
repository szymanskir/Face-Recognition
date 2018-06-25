import numpy as np
import os
import cv2
from math import inf
from natsort import natsorted
from typing import List


def read_image(filepath, equalize = True):
    """
    Reads an image and performs histogram equilization
    depending on arguments
    :param filepath: filepath to image file
    :param equalize: if True perform histogram equalization
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if(equalize):
        image = cv2.equalizeHist(image)

    return image


def read_images(database, equalize):
    """
    Reads images from the given database and performs
    histogram equilization depending on arguments
    :param database: filepath to the database
    :param equalize: if True perform histogram equalization
    """
    faces = list()
    files = natsorted(os.listdir(database))

    for file in files:
        face = read_image(os.path.join(database, file), equalize)
        faces.append(np.array(face, dtype=np.float))

    return faces


# PCA specific functions

def create_face_matrix(faces):
    """
    Constructs a face matrix from given list of faces
    :param faces: list of face images
    """
    face_matrix = list()

    for face in faces:
        face_matrix.append(np.ndarray.flatten(face))
    
    return np.transpose(np.array(face_matrix))


def calculate_pca_eigenvectors(face_matrix):
    """
    Calculates the eigenfaces
    :param face_matrix: matrix of faces
    """
    # compute mean vector
    m = np.mean(face_matrix, 1)

    # calculate centered points
    _, c = face_matrix.shape
    for i in range(0, c):
        face_matrix[:, i] = face_matrix[:, i] - m

    # calculate eigenvalues and eigenvectors
    L = np.dot(np.transpose(face_matrix), face_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(L)

    # sort eigenvalues and eigenvectors
    sorted_indices = eigenvalues.argsort()
    face_space_base = eigenvectors[:,sorted_indices] 

    return face_space_base, m


def get_pca_features(face_filepath, eigenfaces, mean_face):
    """
    Extracts the feature vector from given face
    :param face_filepath: filepath to face image
    :param eigenfaces: matrix of eigenfaces
    :param mean_face: mean face
    """
    face = read_image(face_filepath)
    face = np.ndarray.flatten(face) - mean_face

    return np.dot(np.transpose(eigenfaces), face)


def calculate_database_pca_features(database, eigenfaces, mean_face):
    """
    Extracts feature vectors from all images in a given database
    :param database: filepath to the database
    :param eigenfaces: matrix of eigenfaces
    :param mean_face: mean face
    """
    database_pca_features = list()

    for person in os.listdir(database):
        for sample in os.listdir(os.path.join(database, person)):
            full_path = os.path.join(database, person, sample)
            sample_face_features = get_pca_features(full_path, eigenfaces, mean_face)
            database_pca_features.append({
                'pca_feature': sample_face_features,
                'person' : os.path.join(database, person),
                'path' : os.path.join(database, person, sample)
            })

    return database_pca_features


def identify_face(test_image_filepath, database, eigenfaces, mean_face):
    """
    Identifies the given face in the given database (it is assumed that
    the person is present in the database)
    :param test_image_filepath: filepath to the face image to identify
    :param database: filepath to the detabase
    :param eigenfaces: matrix of eigenfaces
    :param mean_face: mean face
    """
    min_distance = inf
    identified = None
    test_image_features = get_pca_features(test_image_filepath, eigenfaces, mean_face)

    for person in os.listdir(database):
        for sample_face in os.listdir(os.path.join(database, person)):
            sample_face_features = get_pca_features(sample_face, eigenfaces, mean_face)

            current_distance = np.linalg.norm(sample_face_features - test_image_features)
            if min_distance > current_distance:
                min_distance = current_distance
                identified = os.path.join(database, person)

    return identified