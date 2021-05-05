import math
import numpy as np
import random
#import mysklearn.mypytable as mypytable 

#this is used to get the total number of occuances of 
#each diffrent value for a certain column in a table
def get_table_frequency (table, header_id):
    col = table.get_column(header_id)
    ids = []
    values = []
    for val in col:
        if val in ids:
            ind = ids.index(val)
            values[ind] += 1
        else:
            ids.append(val)
            values.append(1)
    return [ids, values]

#this fucntion is given a list of cut offs for a
#column of numerical data and returns how many elements
#were in each group
def range_rating(table, cut_offs, col_name):
    col = table.get_column(col_name)
    values = []
    for row in col:
        for ran in range(len(cut_offs)):
            low = cut_offs[ran][1]
            high = cut_offs[ran][2]
            score = cut_offs[ran][0]
            if ran == 0 and row <= high:
                values.append(score)
            elif ran + 1 == len(cut_offs) and row >= low:
                values.append(score)  
            else:
                if row >= low and row <= high:
                    values.append(score)    
    return values


#this function seprates the elements of a numeric 
#column into five predetemined boxes and returns 
#the number of occuences. 
def five_boxes(table, col_name):
    col = table.get_column(col_name)
    max_val = max(col)
    min_val = min(col)
    difference = max_val - min_val
    division = int(difference/5)
    ranges = []
    start = min_val
    for x in range(5):
        value = [x+1]
        value.append(start)
        value.append(start+division)
        start = start+division+1
        ranges.append(value)
    return range_rating(table,ranges,col_name)


#this fucntion gets the frequency for a set of scores
#and return how many times each score occurs 
def get_range_frequencey(data, scores):
    values = []
    for _ in range(len(scores)):
        values.append(0)
    for row in data:
        if row in scores:
            index = scores.index(row)
            values[index] += 1
    return values

#this computes the slope, intercept, 
#correlation coefficent, and covariance for 
#the input data sets
def compute_slope_intercept(x,y):
    meanX = sum(x)/len(x)
    meanY = sum(y)/len(y)
    m = 0
    top = 0 
    bottomX = 0
    bottomY = 0 
    for i in range(len(x)):
        top +=  ((x[i]-meanX)*(y[i]-meanY))
        bottomX += (x[i]-meanX)**2
        bottomY += (y[i]-meanY)**2
    m = top/bottomX
    b = meanY-m*meanX
    r = top / (bottomX*bottomY)**(1/2)
    cov = top/len(x)
    return m,b,r,cov

#this fucntion gets multiple columns from 
# a data table and returns them to the user
def get_row_totals(table, cols):
    indexes = []
    values = []
    for x in cols:
        indexes.append(table.column_names.index(x))
        values.append(0)
    
    for row in table.data:
        for ind in range(len(indexes)):
            values[ind] += row[indexes[ind]]
    return values

# this fucntion is used to convert ratings into a 
#decimal form in the range of 1 to 10
def convert_to_decimal_rating(ratings):
    fixed = []
    for row in ratings:
        if not row == "":
            temp = str(row).replace("%","")
            temp = float(temp)
            temp = temp%10
            fixed.append(temp)
        else:
            fixed.append("")
    return fixed

#this removes any blank entries from a list
def remove_blanks(data):
    clean =[]
    for x in data:
        if not x == "":
            clean.append(x)
    return clean

#this fucntion returns the average rating for a specified
#column. it returns what the average value for each distinct 
#element of the column is. 
def get_genre_frequency(table, header_id, rate):
    col = table.get_column(header_id)
    ids = []
    values = []
    for val in range(len(col)):
        genre = ""
        rating = rate[val]
        for x in range(len(col[val])):
            if not col[val][x] == ',' and not x == len(col[val]):
                genre += col[val][x]
            else:
                if genre in ids:
                    if not rating == "":
                        ind = ids.index(genre)
                        values[ind] = (values[ind]+float(rating))/2
                else:
                    if not rating == "":
                        ids.append(genre)
                        values.append(float(rating))
                genre = ""
    return [ids, values]

def compute_euclidean_distance(v1, v2):
    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

def compute_mpg_rating(num):
    if num <= 13:
        return 1
    elif num >= 13 and num < 15:
        return 2
    elif num >= 15 and num < 17:
        return 3
    elif num >= 17 and num < 20:
        return 4
    elif num >= 20 and num < 24:
        return 5
    elif num >= 24 and num < 27:
        return 6
    elif num >= 27 and num < 31:
        return 7
    elif num >= 31 and num < 37:
        return 8
    elif num >= 37 and num < 45:
        return 9
    else:
        return 10

def normalize_data(data):
    maximun = max(data)
    mininum = min(data)
    normal = []
    for x in data:
        val = (x - mininum)/(maximun-mininum) * 10.0
        normal.append(val)
    return normal

def get_accuracy_error(matrix):
    accuracy = 0
    error = 0
    total = 0 
    for row in range(len(matrix)):
        total += sum(matrix[row])
        for col in range(len(matrix[row])):
            if row == col:
                accuracy += matrix[row][col]
            else: 
                error += matrix[row][col]
    accuracy = accuracy/total
    error = error/total
    return accuracy, error

def catagorical_to_numeric(data, catagories):
    for row in range(len(data)):
        for col in range(len(data[row])):
            if data[row][col] in catagories:
                index = catagories.index(data[row][col])
                data[row][col] = index
    return data

def catagorical_weight(data):
    result = []
    for x in data:
        if x <= 1999:
            result.append(1)
        elif x>2000 and x<= 2499:
            result.append(2)
        elif x>2500 and x<= 2999:
            result.append(3)
        elif x>3000 and x<= 3499:
            result.append(4)
        else:
            result.append(5)
    return result
        
def print_matrix(label, matrix):
    header = ""
    for i in range(len(label)):
        header += "\t"+str(label[i])
    header += "\tTotal"
    print(header)
    print("---------------------------------------------------------------------------------")
    for x in range(len(matrix)):
        row = str(label[x])+"|"
        for elm in matrix[x]:
            row += "\t"+str(elm)
        row += "\t"+str(sum(matrix[x]))
        print(row)

def make_header(instances):
    header = []
    for col in range(len(instances[0])):
        header.append("att"+str(col))
    return header

def make_att_domain(instances, header):
    dic = {}
    for x in range(len(header)):
        vals = []
        for row in instances:
            if not row[x] in vals:
                vals.append(row[x])
        dic[header[x]] = vals
    return dic

def select_attribute(instances, available_attributes, header, attribute_domain):
    #need to do entropy based selection
    entropy_scores = []
    for att in available_attributes:
        dic = attribute_domain[att]
        row = []
        for _ in range(len(dic)):
            row.append(0)
        values = []
        classes = []
        att_index = header.index(att)
        for element in instances:
            if not element[-1] in classes:
               classes.append(element[-1])
               values.append(row.copy())
            class_ind = classes.index(element[-1])
            val_ind = dic.index(element[att_index])
            values[class_ind][val_ind] += 1
        
        att_entropy = []
        counts = []
        for col in range(len(values[0])):
            total = 0
            vals = []
            entropy = 0
            for row in range(len(values)):
                value = values[row][col]
                total += value
                vals.append(value)
            for x in vals:
                if x > 0:
                    entropy += -(x/total)*(math.log2(x/total))
            att_entropy.append(entropy)
            counts.append(total)  

        new_entropy = 0
        for ind in range(len(att_entropy)):
            new_entropy += (counts[ind]/len(instances))*att_entropy[ind]
        entropy_scores.append(new_entropy)

    index = entropy_scores.index(min(entropy_scores))
    return available_attributes[index]

def all_same_class(instances):
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True

def partition_instances(instances, split_attribute, header, attribute_domains):
    # comments refer to split_attribute "level"
    attribute_domain = attribute_domains[split_attribute] 
    attribute_index = header.index(split_attribute) 

    partitions = {} # key (attribute value): value (partition)
    # task: try to finish this
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions 

def majority_leaf(partition):
    classes = []
    num_classes = []
    for element in partition:
        if not element[-1] in classes:
            classes.append(element[-1])
            num_classes.append(1)
        else:
            index = classes.index(element[-1])
            num_classes[index] += 1
    instances = max(num_classes)
    index = num_classes.index(instances)
    return classes[index], instances

def tdidt(current_instances, available_attributes, header, attribute_domain):
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes,header,attribute_domain)
    #print("splitting on:", split_attribute)

    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domain)
    #print("partitions:", partitions)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        # TODO: append your leaf nodes to this list appropriately
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print("CASE 1")
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            val, instances = majority_leaf(partition)
            #get number of values in the partition. use insted of len(partition)
            leaf = ["Leaf", val, instances, len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            tree = []
            #get all the instances of the higher up attribute and use as partition
            #for the majority leaf and return tree
            val, instances = majority_leaf(current_instances)
            leaf = ["Leaf", val, instances, len(current_instances)]
            return leaf

        else: # all base cases are false, recurse!!
            #print("Recurse")
            #print(available_attributes)
            subtree = tdidt(partition, available_attributes.copy(),header, attribute_domain)
            values_subtree.append(subtree)
            tree.append(values_subtree)         
    return tree

#same as a normal TDIDT but uses random subset of attributes to be chosen from
def random_forest_tdidt(current_instances, available_attributes, header, attribute_domain, F):
    # basic approach (uses recursion!!):
    # select an attribute to split on
    available_attributes = random_attribute_subset(available_attributes, F)
    split_attribute = select_attribute(current_instances, available_attributes,header,attribute_domain)

    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domain)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            val, instances = majority_leaf(partition)
            #get number of values in the partition. use insted of len(partition)
            leaf = ["Leaf", val, instances, len(current_instances)]
            values_subtree.append(leaf)
            tree.append(values_subtree)  

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            tree = []
            #get all the instances of the higher up attribute and use as partition
            #for the majority leaf and return tree
            val, instances = majority_leaf(current_instances)
            leaf = ["Leaf", val, instances, len(current_instances)]
            return leaf

        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(),header, attribute_domain)
            values_subtree.append(subtree)
            tree.append(values_subtree)  
         
    return tree


#this funtion is used by the predict fuction in MyDecisionTree
#it determines which class a single test instance classifies to 
def classify_tdidt(tree, instance):
    prediction = ""
    if tree[0] == "Attribute":
        attribute = tree[1]
        index = int(attribute[-1])
        value = instance[index]
        for x in range(2,len(tree)):
            if tree[x][0] == "Value":
                if tree[x][1] == value:
                    return classify_tdidt(tree[x][2], instance)
    if tree[0] == "Leaf":
        prediction = tree[1]
    return prediction

#this fucnction is used by print_decision_tree_rules in the MyDecisionTree Class
#This determines ther rules of a given tree and prints them to the console
def print_rules(tree, attribute_names, class_name, rules):
    if tree[0] == "Attribute":
        attribute = tree[1]
        index = int(attribute[-1])
        att_name = attribute_names[index]
        if rules == "":
            rules+="IF "
        else:
            rules+=" AND "
        rules += str(att_name) + " == "

        for x in range(2,len(tree)):
            if tree[x][0] == "Value":
                temp = str(tree[x][1])
                print_rules(tree[x][2], attribute_names, class_name, rules+temp)
    if tree[0] == "Leaf":
        rules += " THEN "+ str(class_name) +" = "+str(tree[1])
        print(rules)

def bootstrap_sample(X, y, test=.37, train=.63):
    n = len(X)
    train = math.ceil(n*train)
    test = math.floor(n*test)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []
    for _ in range(train):
        rand_index = random.randrange(train)
        train_set_x.append(X[rand_index])
        train_set_y.append(y[rand_index])
    for _ in range(test):
        rand_index = random.randrange(test)
        test_set_x.append(X[rand_index])
        test_set_y.append(y[rand_index])
    return [train_set_x,train_set_y], [test_set_x,test_set_y]
    
def random_attribute_subset(attributes, F):
    # shuffle and pick first F
    shuffled = attributes[:] # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]

def majority_rule(data):
    classes = []
    num_classes = []
    for element in data:
        if not element == [""]:
            if  not element[-1] in classes:
                classes.append(element[-1])
                num_classes.append(1)
            else:
                index = classes.index(element[-1])
                num_classes[index] += 1
    instances = max(num_classes)
    index = num_classes.index(instances)
    return classes[index]

def normailze_wine_score(score):
    if score >= 95.0:
        return "Excellent"
    elif score >= 90.0:
        return "Great"
    elif score >= 85.0:
        return "Good"
    else:
        return "OK"
    