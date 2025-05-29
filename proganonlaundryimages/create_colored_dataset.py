import os
import change_colors
import cv2
import pickle

if __name__ == '__main__':
    labels = []
    # counter was only added to create a smaller dataset for testing purposes
    #counter = 0
    if not os.path.exists('colored_images'):
        os.mkdir('colored_images')
    for img in os.listdir('dataset'):
        yellow_img = change_colors.change_color(img, 90)
        yellow_img = cv2.resize(yellow_img, (270,180))
        red_img = change_colors.change_color(img, 125)
        red_img = cv2.resize(red_img, (270,180))
        green_img = change_colors.change_color(img, 250)
        green_img = cv2.resize(green_img, (270,180))
        cv2.imwrite(os.path.join('colored_images','yellow_'+img), yellow_img)
        labels.append('yellow')
        cv2.imwrite(os.path.join('colored_images','red_'+img), red_img)
        labels.append('red')
        cv2.imwrite(os.path.join('colored_images','green_'+img), green_img)
        labels.append('green')
        #if counter == 10:
           # break
       # counter += 1

    #pickle.dumps(labels)
