import sys
import getopt
import cv2 as cv
import numpy as np
import re
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from collections import deque

def main():
    argv = sys.argv[1:]
    file_path = ""
    save_path = ""
    n_layers = 0
    max_binary_value = 255
    smoothBoundry = 200

    try:
        opts, args = getopt.getopt(argv, 'f:n:', ['filePath', 'nLayers'])

        if len(opts) == 0 and len(opts) > 2:
            print('usage: add.py -f <file_path> -n <number_of_layers>')
        
        else:
            n_layers = int(opts.pop()[1])
            file_path = opts.pop()[1]
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
                
                main_island_counter, islands_coordinates = countIslands(image, 0)
                if main_island_counter > 1:
                    image_wbridges = connectIslands(image, islands_coordinates)

                alpha = 1 / n_layers
                beta = (1.0 - alpha)
                blended_image = cv.addWeighted(cv.bitwise_not(image), alpha, blended_image, beta, 0.0)
                cv.imwrite(save_path + '\\layers\\layer' + str(i + 1) + '.jpg', image)
                print('layer' + str(i + 1) + ' saved!')

            cv.imwrite(save_path + '\\layers\\blended.jpg', blended_image)
    except getopt.GetoptError:
        print ('usage: add.py -f <file_path> -n <number_of_layers>')
        sys.exit(2)

def smoothUp(image):
    blurred = cv.pyrUp(image)
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
    print('The best state found is: ', best_state)

    image_copy = image
    return image_copy

if __name__ == "__main__":
    main()