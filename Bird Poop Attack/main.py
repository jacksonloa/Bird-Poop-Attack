import cv2 as cv
import easyocr
import os
import sys
import numpy as np
import random
from util import img_process, crop_lp, character_recognize, lp_recognize, check_all_same
from attack import genetic_algorithm, generate_random, evaluate_fitness
from collections import defaultdict, Counter

reader = easyocr.Reader(['en'])

img_folder = 'data/img'
label_folder = 'label'
pos_folder = 'pos'
adv_folder = 'adv_test'
wrong_list_path = 'wrong_list.txt'

total = 0
correct = 0
accuracy = 0


for filename in os.listdir(img_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 讀取圖片
        img_path = os.path.join(img_folder, filename)
        label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
        pos_path = os.path.join(pos_folder, os.path.splitext(filename)[0] + '.txt')
        print('img : ', img_path)
     
        total += 1
        if os.path.exists(img_path):
            # 讀取圖片，處理圖片（調整大小，裁切圖片）
            img = cv.imread(img_path)
           
            cv.namedWindow('window', cv.WINDOW_NORMAL)
            cv.resizeWindow('window', 800, 600)
            cv.imshow('window', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            # 調整大小，裁切圖片剩車牌
            crop_img = crop_lp(img) 
            h, w, _ = crop_img.shape

            cv.namedWindow('window', cv.WINDOW_NORMAL)
            cv.resizeWindow('window', 800, 600)
            cv.imshow('window', crop_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

            # 對圖片做預處理，進行影像辨識
            pimg = img_process(crop_img)
            lp_result = lp_recognize(pimg)

            # 讀取標籤文件
            with open(label_path, 'r') as file:
                label = file.read()

            
            if lp_result != label:
                print('車牌辨識結果錯誤，無須進行攻擊')
                continue

            # 將圖片依照各個車牌號碼位置，將車牌圖片分為六張新的圖片
            characters = []
            characters_pos = []
            with open(pos_path, 'r') as pos_file:
                pos_data = pos_file.readlines()
    
                for line in pos_data:
                    if(line == '\n'):
                        print('pass')
                        continue
                    x1, x2 = map(int, line.split())
                    y1 = 0
                    y2 = h
                    character = crop_img[y1:y2, x1:x2]
                    characters.append(character)
                    characters_pos.append((x1, x2))

            # 利用easyocr選擇欲攻擊的字（選取正確但信心值較小值的車牌號碼）
            # 除存target character與character的位置
            tar_idx = -1
            min_conf = 1
            target_character = ''
        
            for idx, img in enumerate(characters):
                # 一樣先做影像處理
                cleaned_thresh = img_process(img)
                # 做影像辨識
                result = reader.readtext(cleaned_thresh, detail=1)
                # print(result)
                # print('label :', label[idx])
                # 選出信心值最小的車牌
                # 不選難以攻擊的字元
                hard_array = ['N', '1', 'I']
                hard_to_attack = 0

                for info in result:
                    if any(info[1] == char for char in hard_array):
                        hard_to_attack = 1

                    if info[1].isdigit() or info[1].isalpha():
                        if hard_to_attack != 1:
                            if info[2] < min_conf:
                                tar_idx = idx
                                min_conf = info[2]
                                target_character = info[1]
            print(target_character)
            print(tar_idx)
            print(min_conf)

            cv.namedWindow('window', cv.WINDOW_NORMAL)
            cv.resizeWindow('window', 800, 600)
            cv.imshow('window', characters[tar_idx])
            cv.waitKey(0)
            cv.destroyAllWindows()


            # 定義genetic_algorithm迭代的次數
            iterations = 20

            # 初始化种群
            population_size = 20
            center_std_dev = 10  # 中心點的標準差
            std_dev = 3  # 標準差

            original_image = np.array(characters[tar_idx])


            # 生成250個隨機位置的擾動
            num_points = 250
            found_target_label = 0
            found_cnt = 0

            move_left = 0
            move_right = 0
            while found_target_label == 0:
                if found_cnt > 35:
                    print('Generation超過40次 更換target character')
                 
                    target1_idx = -1
                    target2_idx = -1
                    
                    if tar_idx >= 3 and move_left == 0 and move_right == 0:
                        move_left = 1
                        tar_idx -= 1
                        p_img = img_process(characters[tar_idx])
                        next_result = character_recognize(p_img)
                        for text, prob in next_result:
                            target_character = text

                        if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                            original_image = np.array(characters[tar_idx])
                            found_cnt = 0
                            print('將character改為 ', target_character)
                            continue
                        else:
                            tar_idx -= 1
                            p_img = img_process(characters[tar_idx])
                            next_result = character_recognize(p_img)
                            for text, prob in next_result:
                                target_character = text
                                
                            if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                original_image = np.array(characters[tar_idx])
                                found_cnt = 0
                                print('將character改為 ', target_character)
                                continue
                            else:
                                tar_idx -= 1
                                p_img = img_process(characters[tar_idx])
                                next_result = character_recognize(p_img)
                                for text, prob in next_result:
                                    target_character = text

                                if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                    original_image = np.array(characters[tar_idx])
                                    found_cnt = 0
                                    print('將character改為 ', target_character)
                                else:
                                    print('generation failed')
                                    break
                    elif tar_idx < 3 and move_left == 0 and move_right == 0:
                        move_right = 1
                        tar_idx += 1
                        p_img = img_process(characters[tar_idx])
                        next_result = character_recognize(p_img)
                      
                        for text, prob in next_result:
                            target_character = text

                        if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                            original_image = np.array(characters[tar_idx])
                            found_cnt = 0
                            print('將character改為 ', target_character)
                            continue
                        else:
                            tar_idx += 1
                            p_img = img_process(characters[tar_idx])
                            next_result = character_recognize(p_img)
                            for text, prob in next_result:
                                target_character = text

                            if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                original_image = np.array(characters[tar_idx])
                                found_cnt = 0
                                print('將character改為 ', target_character)
                                continue
                            else:
                                tar_idx += 1
                                p_img = img_process(characters[tar_idx])
                                next_result = character_recognize(p_img)
                                for text, prob in next_result:
                                    target_character = text

                                if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                    original_image = np.array(characters[tar_idx])
                                    found_cnt = 0
                                    print('將character改為 ', target_character)
                                    continue
                                else:
                                    print('generation failed')
                                    break

                    elif move_left == 1 or move_right == 1:
                        if move_left == 1:
                            if tar_idx-1 >= 0:
                                
                                tar_idx -= 1
                                p_img = img_process(characters[tar_idx])
                                next_result = character_recognize(p_img)
                                for text, prob in next_result:
                                    target_character = text

                                if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                    original_image = np.array(characters[tar_idx])
                                    found_cnt = 0
                                    print('將character改為 ', target_character)
                                    continue
                                else:
                                    if tar_idx-1 >= 0:
                                        tar_idx -= 1
                                        p_img = img_process(characters[tar_idx])
                                        next_result = character_recognize(p_img)
                                        for text, prob in next_result:
                                            target_character = text
                                            
                                        if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                            original_image = np.array(characters[tar_idx])
                                            found_cnt = 0
                                            print('將character改為 ', target_character)
                                            continue
                                        else:
                                            if tar_idx-1 >= 0:
                                                tar_idx -= 1
                                                p_img = img_process(characters[tar_idx])
                                                next_result = character_recognize(p_img)
                                                for text, prob in next_result:
                                                    target_character = text

                                                if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                                    original_image = np.array(characters[tar_idx])
                                                    found_cnt = 0
                                                    print('將character改為 ', target_character)
                                                else:
                                                    print('generation failed')
                                                    break
                                            else:
                                                print('generation failed')
                                                break
                                    else:
                                        print('generation failed')
                                        break
                            else:
                                print('generation failed')
                                break


                        elif move_right == 1:
                            if tar_idx+1 <= 5:
                                
                                tar_idx += 1
                                p_img = img_process(characters[tar_idx])
                                next_result = character_recognize(p_img)
                                for text, prob in next_result:
                                    target_character = text

                                if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                    original_image = np.array(characters[tar_idx])
                                    found_cnt = 0
                                    print('將character改為 ', target_character)
                                    continue
                                else:
                                    if tar_idx+1 <= 5:
                                        tar_idx += 1
                                        p_img = img_process(characters[tar_idx])
                                        next_result = character_recognize(p_img)
                                        for text, prob in next_result:
                                            target_character = text
                                            
                                        if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                            original_image = np.array(characters[tar_idx])
                                            found_cnt = 0
                                            print('將character改為 ', target_character)
                                            continue
                                        else:
                                            if tar_idx+1 <= 5:
                                                tar_idx += 1
                                                p_img = img_process(characters[tar_idx])
                                                next_result = character_recognize(p_img)
                                                for text, prob in next_result:
                                                    target_character = text

                                                if target_character not in hard_array and label[tar_idx] not in hard_array and target_character:
                                                    original_image = np.array(characters[tar_idx])
                                                    found_cnt = 0
                                                    print('將character改為 ', target_character)
                                                else:
                                                    print('generation failed')
                                                    break
                                            else:
                                                print('generation failed')
                                                break
                                    else:
                                        print('generation failed')
                                        break
                            else:
                                print('generation failed')
                                break
                        

                population = []
                population = generate_random(original_image, num_points, population_size, center_std_dev, std_dev)

                # 儲存結果用
                result_list = []
               
                for img in population:
                    pimg = img_process(img)
                    result = character_recognize(pimg)
                    result_list.append(result)

                positions = defaultdict(list)

                for i, result in enumerate(result_list):
                    for text, prob in result:
                        positions[text].append(i)

                # 檢查是否有text元素重複出現並顯示其位置
                duplicates = []
                duplicates = [(text, pos) for text, pos in positions.items() if len(pos) > 1]

                # 不選因為相似而判斷錯誤的車牌號碼
                similar_array =[('8', 'B'), ('0', 'D'), ('0', 'Q'), ('S', '8'), ('Y', 'V'), ('Z', '2')]
                similar = 0

                if duplicates:
                    target_pos = []
                    target_text = []

                    for text, pos in duplicates:
                        for c1, c2 in similar_array:
                            if (text == c1 and target_character == c2) or (text == c2 and target_character == c1) or (text == c1 and label[tar_idx] == c2) or (text == c2 and label[tar_idx] == c1):
                                similar = 1
                        if text != label[tar_idx] and (text.isdigit() or text.isalpha()) and similar != 1:
                            print(f"元素 {text} 重複出現在位置 {pos}")
                            target_pos.append(pos)
                            target_text.append(text)
                            found_target_label = 1
                        else:
                            print('target class不符合標準')
                else:
                    print("沒有text元素重複出現。")

                if found_target_label == 0:
                    print('沒找到重複字元')
                    found_cnt += 1
                    continue

                # 目標是抓取兩張同樣character且confidence高的圖片
                target1_confidence = 0
                target2_confidence = 0
                target1_idx = -1
                target2_idx = -1
                regenerate = 0
                list_pos = 0
                tc = ''

                for pos in target_pos:
                    regenerate = 0
                    for i in pos:
                        for text, prob in result_list[i]:
                            if prob > target1_confidence:
                                if target1_confidence > target2_confidence:
                                    target2_confidence = target1_confidence
                                    target2_idx = target1_idx
                                target1_confidence = prob
                                target1_idx = i
                            elif prob > target2_confidence:
                                target2_confidence = prob
                                target2_idx = i

                    if max(target1_confidence, target2_confidence) < 0.7 or min(target1_confidence, target2_confidence) < 0.5 or max(target1_confidence, target2_confidence) - min(target1_confidence, target2_confidence) > 0.5 :
                        print('太小或差太多')
                        target1_confidence = 0
                        target2_confidence = 0
                        target1_idx = -1
                        target2_idx = -1
                        regenerate = 1
                        list_pos += 1
                        continue

                    print('found target')
                    tc = target_text[list_pos]
                    print(tc, target1_idx, target1_confidence, target2_idx, target2_confidence)

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', population[target1_idx])
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', population[target2_idx])
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                
                    break
                if regenerate == 1:
                    found_target_label = 0
                    found_cnt += 1
                    print('重作圖片')

            if target1_idx == -1 or target2_idx == -1:
                print('Attack Failed')
                continue

            group = []
            group.append(population[target1_idx])
            group.append(population[target2_idx])
            mutation_rate = 0.1

            t_x1 = characters_pos[tar_idx][0]
            t_x2 = characters_pos[tar_idx][1]
            t_y1 = 0
            t_y2 = h
            oc = label[tar_idx]

            # 利用generic algorithm產生對抗樣本
            final_group, found = genetic_algorithm(group, tc, oc, iterations, population_size, mutation_rate, crop_img, t_x1, t_x2, t_y1, t_y2, tar_idx)
            print('GA finish')

            pixel_changes = Counter()

            # 找出final group共同有被修改過的pixel
            if found == 1:
                print('found = 1')
                check_lp_word = []
                check_lp_fitness = []
                check_c_word = []
                check_c_fitness = []
                for img in final_group:
                    check_lp_img = crop_img
                    check_lp_img[t_y1:t_y2, t_x1:t_x2] = img
                    p_lp_img = img_process(check_lp_img)
                    check_lp_result = lp_recognize(p_lp_img)
                    p_c_img = img_process(img)
                    check_c_result = character_recognize(p_c_img)
                    
                    check_lp_word.append(check_lp_result[tar_idx])
                    
                    for text, prob in check_c_result:
                        check_c_word.append(text)
                        check_c_fitness.append(prob)

                if check_all_same(check_lp_word):
                    print('all the same')
                    for image in final_group:
                        # 计算与原始图像的差异
                        difference = image - original_image

                        # 找到所有非零的像素
                        changed_pixels = np.transpose(np.nonzero(difference))

                        # 将每个变化的像素添加到Counter中
                        for pixel in changed_pixels:
                            pixel_changes[tuple(pixel)] += 1
                        print('changed pixel ', len(changed_pixels))

                    # 找出出现次数最多的前250个像素
                    top_250_changed_pixels = pixel_changes.most_common(300)
                    p_result = [((x, y, _), count) for ((x, y, _), count) in top_250_changed_pixels]
                    p_result = [((x, y)) for ((x, y, _), _) in p_result]

                    random_choices = random.sample(range(len(p_result)), 5)
                    random_pos = np.array([p_result[pos] for pos in random_choices])

                    perturbed_img = original_image
                    for idx, (x, y) in enumerate(p_result):
                        if any((random_pos[:, 0] == x) & (random_pos[:, 1] == y)):
                            perturbed_img[x, y] = [136, 184, 201]
                            if y+1 < perturbed_img.shape[1]:
                                perturbed_img[x, y+1] = [161, 209, 187]
                            continue
                        original_value = original_image[x, y] 
                        perturbation = np.array([90, 90, 90])
                        new_value = np.clip(original_value + np.array(perturbation), 0, 255)               
                        perturbed_img[x, y] = new_value

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', original_image)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    crop_img[t_y1:t_y2, t_x1:t_x2] = perturbed_img

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', crop_img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    process_img = img_process(crop_img)
                    final_result = lp_recognize(process_img)

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', process_img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    if final_result == label:
                        pixel_changes = Counter()
                        difference = final_group[0] - original_image

                        # 找到所有非零的像素
                        changed_pixels = np.transpose(np.nonzero(difference))

                        # 将每个变化的像素添加到Counter中
                        for pixel in changed_pixels:
                            pixel_changes[tuple(pixel)] += 1
                        

                        # 找出出现次数最多的前250个像素
                        top_250_changed_pixels = pixel_changes.most_common(300)
                        p_result = [((x, y, _), count) for ((x, y, _), count) in top_250_changed_pixels]
                        p_result = [((x, y)) for ((x, y, _), _) in p_result]
                        
                        random_choices = random.sample(range(len(p_result)), 5)
                        random_pos = np.array([p_result[pos] for pos in random_choices])

                        perturbed_img = original_image
                        for idx, (x, y) in enumerate(p_result):
                            if any((random_pos[:, 0] == x) & (random_pos[:, 1] == y)):
                                perturbed_img[x, y] = [136, 184, 201]
                                if y+1 < perturbed_img.shape[1]:
                                    perturbed_img[x, y+1] = [161, 209, 187]
                                continue
                            original_value = original_image[x, y] 
                            perturbation = np.array([90, 90, 90])
                            new_value = np.clip(original_value + np.array(perturbation), 0, 255)
                            perturbed_img[x, y] = new_value

                        crop_img[t_y1:t_y2, t_x1:t_x2] = perturbed_img
                        process_img = img_process(crop_img)
                        final_result = lp_recognize(process_img)

                        if final_result == label:
                            perturbed_img = original_image
                            for idx, (x, y) in enumerate(p_result):
                                perturbed_img[x, y] = (255, 255, 255)

                            crop_img[t_y1:t_y2, t_x1:t_x2] = perturbed_img
                            process_img = img_process(crop_img)
                            last_result = lp_recognize(process_img)

                            if last_result == label:
                                print("Attack Failed")
                                with open(wrong_list_path, 'a') as file:
                                    file.write(f"img: {img_path}\n")
                                    file.write(f"label[{tar_idx}]: {label[tar_idx]}\n")
                                    file.write(f"target_character: {target_character}\n")
                                    file.write(f"tc: {tc}\n")
                                    file.write("\n")

                                continue
                            else:
                                print("Attack Success")
                                adv_path = adv_path = os.path.join(adv_folder, filename)
                                cv.imwrite(adv_path, crop_img)
                                correct += 1
                                continue
                        else:  
                            print("Attack Success")
                            adv_path = adv_path = os.path.join(adv_folder, filename)
                            cv.imwrite(adv_path, crop_img)
                            correct += 1
                            continue
                    else:
                        print("Attack Success")
                        adv_path = adv_path = os.path.join(adv_folder, filename)
                        cv.imwrite(adv_path, crop_img)
                        correct += 1
                        continue

                else:
                    print('different')
                    max_index = np.argmax(check_c_fitness)
                    difference = final_group[max_index] - original_image

                    # 找到所有非零的像素
                    changed_pixels = np.transpose(np.nonzero(difference))

                    # 将每个变化的像素添加到Counter中
                    for pixel in changed_pixels:
                        pixel_changes[tuple(pixel)] += 1

                    # 找出出现次数最多的前250个像素
                    top_250_changed_pixels = pixel_changes.most_common(300)
                    p_result = [((x, y, _), count) for ((x, y, _), count) in top_250_changed_pixels]
                    p_result = [((x, y)) for ((x, y, _), _) in p_result]

                    random_choices = random.sample(range(len(p_result)), 5)
                    random_pos = np.array([p_result[pos] for pos in random_choices])

                    perturbed_img = original_image       
                    for idx, (x, y) in enumerate(p_result):
                        if any((random_pos[:, 0] == x) & (random_pos[:, 1] == y)):
                            perturbed_img[x, y] = [136, 184, 201]
                            if y+1 < perturbed_img.shape[1]:
                                perturbed_img[x, y+1] = [161, 209, 187]
                            continue
                        original_value = original_image[x, y] 
                        perturbation = np.array([90, 90, 90])
                        new_value = np.clip(original_value + np.array(perturbation), 0, 255)
                        perturbed_img[x, y] = new_value
               

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', original_image)
                    cv.waitKey(0)
                    cv.destroyAllWindows()


                    crop_img[t_y1:t_y2, t_x1:t_x2] = perturbed_img

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', crop_img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    process_img = img_process(crop_img)
                    final_result = lp_recognize(process_img)

                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', process_img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                    if final_result == label:
                        print("Attack Failed")
                        with open(wrong_list_path, 'a') as file:
                            file.write(f"img: {img_path}\n")
                            file.write(f"label[{tar_idx}]: {label[tar_idx]}\n")
                            file.write(f"target_character: {target_character}\n")
                            file.write(f"tc: {tc}\n")
                            file.write("\n")

                        continue
                    else:
                        print("Attack Success")
                        adv_path = adv_path = os.path.join(adv_folder, filename)
                        cv.imwrite(adv_path, crop_img)
                        correct += 1
                        continue
                        

            else:
                print('found 0')
                final_group, final_fitness, found = evaluate_fitness(final_group, tc, oc, crop_img, t_x1, t_x2, t_y1, t_y2, tar_idx)
                
                print(len(final_group))
                for img in final_group:
                    cv.namedWindow('window', cv.WINDOW_NORMAL)
                    cv.resizeWindow('window', 800, 600)
                    cv.imshow('window', img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                
                if len(final_group) == 0:
                    print("Attack Failed")
                    with open(wrong_list_path, 'w') as file:
                        file.write(f"label[{tar_idx}]: {label[tar_idx]}\n")
                        file.write(f"target_character: {target_character}\n")
                        file.write(f"tc: {tc}\n")

                    continue

                for image in final_group:
                    # 计算与原始图像的差异
                    difference = image - original_image

                    # 找到所有非零的像素
                    changed_pixels = np.transpose(np.nonzero(difference))

                    # 将每个变化的像素添加到Counter中
                    print(len(changed_pixels))
                    for pixel in changed_pixels:
                        pixel_changes[tuple(pixel)] += 1


                # 找出出现次数最多的前250个像素
                top_250_changed_pixels = pixel_changes.most_common(300)
                p_result = [((x, y, _), count) for ((x, y, _), count) in top_250_changed_pixels]
                p_result = [((x, y)) for ((x, y, _), _) in p_result]

                random_choices = random.sample(range(len(p_result)), 5)
                random_pos = np.array([p_result[pos] for pos in random_choices])

                perturbed_img = original_image
                for idx, (x, y) in enumerate(p_result):
                    if any((random_pos[:, 0] == x) & (random_pos[:, 1] == y)):
                        perturbed_img[x, y] = [136, 184, 201]
                        if y+1 < perturbed_img.shape[1]:
                                perturbed_img[x, y+1] = [161, 209, 187]
                        continue
                    original_value = original_image[x, y] 
                    perturbation = np.array([90, 90, 90])
                    new_value = np.clip(original_value + np.array(perturbation), 0, 255)
                    perturbed_img[x, y] = new_value
                   

                cv.namedWindow('window', cv.WINDOW_NORMAL)
                cv.resizeWindow('window', 800, 600)
                cv.imshow('window', original_image)
                cv.waitKey(0)
                cv.destroyAllWindows()

        
                crop_img[t_y1:t_y2, t_x1:t_x2] = perturbed_img

                cv.namedWindow('window', cv.WINDOW_NORMAL)
                cv.resizeWindow('window', 800, 600)
                cv.imshow('window', crop_img)
                cv.waitKey(0)
                cv.destroyAllWindows()

                process_img = img_process(crop_img)
                final_result = lp_recognize(process_img)

                cv.namedWindow('window', cv.WINDOW_NORMAL)
                cv.resizeWindow('window', 800, 600)
                cv.imshow('window', process_img)
                cv.waitKey(0)
                cv.destroyAllWindows()

                if final_result == label:
                    pixel_changes = Counter()
                    difference = final_group[0] - original_image

                    # 找到所有非零的像素
                    changed_pixels = np.transpose(np.nonzero(difference))

                    # 将每个变化的像素添加到Counter中
                    for pixel in changed_pixels:
                        pixel_changes[tuple(pixel)] += 1
                    crop_img[t_y1:t_y2, t_x1:t_x2] = final_group[0]

                    # 找出出现次数最多的前250个像素
                    top_250_changed_pixels = pixel_changes.most_common(300)
                    p_result = [((x, y, _), count) for ((x, y, _), count) in top_250_changed_pixels]
                    p_result = [((x, y)) for ((x, y, _), _) in p_result]

                    random_choices = random.sample(range(len(p_result)), 5)
                    random_pos = np.array([p_result[pos] for pos in random_choices])

                    perturbed_img = original_image                       
                        
                    for idx, (x, y) in enumerate(p_result):
                        if any((random_pos[:, 0] == x) & (random_pos[:, 1] == y)):
                            perturbed_img[x, y] = [136, 184, 201]
                            if y+1 < perturbed_img.shape[1]:
                                perturbed_img[x, y+1] = [161, 209, 187]
                            continue
                        original_value = original_image[x, y] 
                        perturbation = np.array([90, 90, 90])
                        new_value = np.clip(original_value + np.array(perturbation), 0, 255)
                        perturbed_img[x, y] = new_value

                    crop_img[t_y1:t_y2, t_x1:t_x2] = original_image
                    process_img = img_process(crop_img)
                    final_result = lp_recognize(process_img)
                    if final_result == label:
                        print("Attack Failed")
                        with open(wrong_list_path, 'a') as file:
                            file.write(f"img: {img_path}\n")
                            file.write(f"label[{tar_idx}]: {label[tar_idx]}\n")
                            file.write(f"target_character: {target_character}\n")
                            file.write(f"tc: {tc}\n")
                            file.write("\n")

                        continue
                    else:  
                        print("Attack Success")
                        adv_path = adv_path = os.path.join(adv_folder, filename)
                        cv.imwrite(adv_path, crop_img)
                        correct += 1
                        continue

                else:
                    print("Attack Success")
                    adv_path = adv_path = os.path.join(adv_folder, filename)
                    cv.imwrite(adv_path, crop_img) 
                    correct += 1
                    continue    

accuracy = (float)(correct/total)  
print(accuracy)     




    
    










#    



# # 在高度重複的區域，利用鳥屎生成演算法生成鳥屎

# # 將產出的20張對抗性車牌號碼圖片貼回去照片的相同位置

# # 在利用easyocr去辨識對抗樣本的結果

# # 分析錯誤結果中車牌號碼照片在哪裡容易被攻擊，即擾動處高度重疊
# #
# #
# #
# #
# #
# #
# #




