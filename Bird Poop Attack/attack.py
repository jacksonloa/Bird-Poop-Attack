import os
import numpy as np
import easyocr
import random
import cv2 as cv
from util import img_process, crop_lp, character_recognize, lp_recognize

reader = easyocr.Reader(['en'])

def get_target_class(file_path):
    with open(file_path, 'r') as file:
        target_class = file.read()
    return target_class


def generate_random(img, num_points, population_size, center_std_dev, std_dev):
    random_img = []
    h, w, _ = img.shape
    for _ in range(population_size):
        center = (int(np.random.normal(w // 2, center_std_dev)), int(np.random.normal(h // 2, center_std_dev)))
        distorted_img = np.zeros_like(img, dtype=np.uint8)
        for _ in range(num_points):
            x = int(np.random.normal(center[0], std_dev))
            y = int(np.random.normal(center[1], std_dev))
   
            value1 = 255
            value2 = 255
            value3 = 255
            # 確保位置在圖片範圍內
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            distorted_img[y, x] = [value1, value2, value3]  # 對 BGR 三個通道同時賦值
        adv_img = cv.add(img, distorted_img)
        random_img.append(adv_img)
    return random_img

# 記錄是否找到成功的對抗樣本


def evaluate_fitness(population, target_class, original_class, lp_img, c_x1, c_x2, c_y1, c_y2, c_pos):
    # 對每個樣本利用Easyocr計算信心值，確保結果不等於原始class且等於目標class
    fitness = []
    new_population = []
    adv_example = []
    adv_fitness = []
    found = 0
    for img in population:
        pimg = img_process(img)
        result = character_recognize(pimg)

        for detection in result:
            if detection[0] != original_class and detection[0] == target_class:
                lp_img[c_y1:c_y2, c_x1:c_x2] = img
                process_img = img_process(lp_img)
                lp_result = lp_recognize(process_img)
                if len(lp_result) != 6 and len(population) != 2:
                    break
                if len(lp_result) == 6:
                    if lp_result[c_pos] != original_class:
                        adv_fitness.append(detection[1])
                        adv_example.append(img)
                        found = 1
            
                        continue
                fitness.append(detection[1])
                new_population.append(img)
    if found == 1:
        print(found)
        return adv_example, adv_fitness, found
    return new_population, fitness, found


def select_parents(population, fitness, num_parents):
    #將population排序後選出前num_parents個的parents
    parents = []
    cnt = 0
    sorted_data = sorted(zip(fitness, population), key=lambda x: x[0], reverse=True)
    # sorted_fitness = [item[0] for item in sorted_data]
    sorted_group = [item[1] for item in sorted_data]
    for img in sorted_group:
        if cnt <= num_parents:
            parents.append(img)
            cnt += 1     
    return parents


def crossover(parents, population_size):
    # 透過parents產生出與之有相似特徵的後代
    offsprings = []
    for _ in range(population_size):
        parent1, parent2 = random.sample(parents, 2)
        mask = np.random.choice([0, 1], size=parent1.shape)
        offspring = parent1 * mask + parent2 * (1 - mask)
        offspring = offspring.astype(np.uint8)
        offsprings.append(offspring)

    return offsprings

def mutate(images, mutation_rate):
    # 讓後代自己產生變異
    mutated_images = []
    for image in images:
        mask = np.random.choice([0, 1], size=image.shape, p=[1 - mutation_rate, mutation_rate])
        mutation = np.random.normal(1, 0.01, size=image.shape)
        mutated_image = image + mask * mutation
        mutated_image = mutated_image.astype(np.uint8)
        mutated_images.append(mutated_image)
        
    return mutated_images

def genetic_algorithm(population, target_class, original_calss, iterations, population_size, mutation_rate, lp_img, c_x1, c_x2, c_y1, c_y2, c_pos):
    
    first_time = 0
    for _ in range(iterations):
        if first_time == 1:
            new_group, fitness, found = evaluate_fitness(group, target_class, original_calss, lp_img, c_x1, c_x2, c_y1, c_y2, c_pos)
        else:
            new_group, fitness, found = evaluate_fitness(population, target_class, original_calss, lp_img, c_x1, c_x2, c_y1, c_y2, c_pos)
        if found == 1:
            return new_group, found
        parents = select_parents(new_group, fitness, population_size // 2)
        if len(parents) < 2:
            return parents, found
        offsprings = crossover(parents, population_size)
        mutate_images = mutate(offsprings, mutation_rate)
        group = parents + mutate_images
    
    return group, found




