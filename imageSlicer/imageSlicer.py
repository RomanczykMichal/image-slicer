import sys
import getopt
import cv2 as cv
import numpy as np
import re
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import imutils
import numpy2stl as n2s
from collections import deque
from skimage.metrics import structural_similarity as compare_ssim
from stl import mesh

'''
    todo:
    - naprawic laczenie mostow
    - ogarnac czemu stl rysuje tylko gorna warstwe
'''

def main():
    argv = sys.argv[1:]
    file_path = ""
    save_path = ""
    n_layers = 0
    max_binary_value = 255
    smoothBoundry = 200

    try:
        opts, args = getopt.getopt(argv, 'f:n:b:s:', ['filePath', 'nLayers', 'bridges', 'stls'])

        if len(opts) == 0 and len(opts) > 2:
            print('usage: add.py -f <file_path> -n <number_of_layers> -b <bridges 1-yes, 0-no> -s <generate stl 1-yes, 0-no>')
        
        else:
            file_path = opts[0][1]
            n_layers = int(opts[1][1])
            do_bridges = int(opts[2][1])
            do_stl = int(opts[3][1])
            regexed = re.search('(.*)\\\\.*\.jpg', file_path)
            if regexed:
                save_path = regexed.group(1)

            src_image = cv.imread(file_path)
            src_image_gray = cv.cvtColor(src_image, cv.COLOR_BGR2GRAY)
            blended_image = np.zeros((len(src_image),len(src_image[0]),), dtype=np.uint8)

            for i in range(n_layers):
                print('preparing layer' + str(i + 1) + '.')
                layer_brightness_level = np.floor(max_binary_value/(n_layers)) * (i + 1)
                _, image = cv.threshold(src_image_gray, layer_brightness_level, max_binary_value, 0)
                
                if (len(image) >= smoothBoundry and len(image[0]) >= smoothBoundry):
                    image = smoothUp(image)

                image = createBorder(image)
                
                if do_bridges == 1:
                    main_island_counter, islands_coordinates = countIslands(image, 0)
                
                    if main_island_counter > 1:
                        image_wbridges, image_bridges = connectIslands(image, islands_coordinates)
                        cv.imwrite(save_path + '\\layers\\layer' + str(i + 1) + '.jpg', image_wbridges)
                        cv.imwrite(save_path + '\\layers\\layer' + str(i + 1) + 'bridge.jpg', image_bridges)
                        print('layer' + str(i + 1) + ' with bridge picture saved!')
                                        
                        if (do_stl == 1):
                            createStf(image_wbridges, save_path + '\\stls\\layer'+str(i + 1)+'Stl.stl')
                            createStf(image_bridges, save_path + '\\stls\\layer'+str(i + 1)+'BrigdeStl.stl')

                    else: 
                        cv.imwrite(save_path + '\\layers\\layer' + str(i + 1) + '.jpg', image)
                        print('layer' + str(i + 1) + ' saved!')

                        if (do_stl == 1):
                            createStf(image, save_path + '\\stls\\layer'+str(i + 1)+'Stl.stl')
                else:
                    cv.imwrite(save_path + '\\layers\\layer' + str(i + 1) + '.jpg', image)
                    print('layer' + str(i + 1) + ' saved!')

                    if (do_stl == 1):
                        createStf(image, save_path + '\\stls\\layer'+str(i + 1)+'Stl.stl')


                alpha = 1 / n_layers
                beta = (1.0 - alpha)
                blended_image = cv.addWeighted(cv.bitwise_not(image), alpha, blended_image, beta, 0.0)
                print()

            cv.imwrite(save_path + '\\layers\\blended.jpg', blended_image)
    except getopt.GetoptError:
        print ('getopt.GetoptError')
        sys.exit(2)

def smoothUp(image):
    blurred = cv.pyrUp(image)
    blurred = cv.medianBlur(blurred, 7)
    blurred = cv.medianBlur(blurred, 7)
    blurred = cv.pyrDown(blurred)
    _, smooth = cv.threshold(blurred,200, 255, cv.THRESH_BINARY_INV)
    return smooth

def createBorder(image):
    row_size = len(image)
    col_size = len(image[0])
    for i in range(row_size):
        if i in (0,1, row_size - 2, row_size - 1):
            for j in range(col_size):
                image[i][j] = 255
        else:
            image[i][0] = 255
            image[i][1] = 255
            image[i][col_size-2] = 255
            image[i][col_size-1] = 255
    return image

def isSafe(mat, i, j, vis):     
    return ((i >= 0) and (i < len(mat)) and 
            (j >= 0) and (j < len(mat[0])) and
            (mat[i][j] and (not vis[i][j])))
 
def BFS(mat, vis, si, sj):
    row = [-1, -1, -1, 0, 0, 1, 1, 1]
    col = [-1, 0, 1, -1, 1, -1, 0, 1]
 
    q = deque()
    q.append([si, sj])
    vis[si][sj] = True

    x = sj
    y = si
 
    while (len(q) > 0):
        temp = q.popleft()
 
        i = temp[0]
        j = temp[1]
 
        for k in range(8):
            if (isSafe(mat, i + row[k], j + col[k], vis)):
                vis[i + row[k]][j + col[k]] = True
                x = j + col[k]
                y = i + row[k]
                q.append([i + row[k], j + col[k]])
    return (x, y)
 
def countIslands(mat, val):
    islands_coordinates = []
    vis = [[False for i in range(len(mat[0]))] for i in range(len(mat))]
    res = 0
 
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if (mat[i][j] and not vis[i][j]):
                x, y = BFS(mat, vis, i, j)
                islands_coordinates.append((x, y))
                res += 1
    return res, islands_coordinates


def connectIslands(image, islands_coords):
    fitness_coords = mlrose.TravellingSales(coords = islands_coords)
    problem_fit = mlrose.TSPOpt(length = len(islands_coords), fitness_fn = fitness_coords, maximize=False)
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 2)

    line_thickness = 2
    image_wbridges = image.copy()
    for i in range(1, len(best_state)):
        p1 = islands_coords[best_state[i-1]]
        p2 = islands_coords[best_state[i]]
        cv.line(image_wbridges, p1, p2, 255,thickness = line_thickness)

    diff = cv.absdiff(image, image_wbridges)
    diff = cv.bitwise_not(diff)

    return image_wbridges, diff

def createStf(file_to_save, file_name):
    n2s.numpy2stl(file_to_save, file_name)

if __name__ == "__main__":
    main()