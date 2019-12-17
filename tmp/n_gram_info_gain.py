import malware_classification.global_var as GLVAR
import numpy as np
import json
from sklearn.model_selection import train_test_split
import copy
import re
import math
from tmp.InformationGain import InformationGain


def generate_ngrams(s, n):
    # Convert to lowercases
    s = s.lower()

    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    # return list demo:"aa-bb-cc bb-cc-dd ..."
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return ["-".join(ngram) for ngram in ngrams]


def create_n_gramed_data(raw_api_filename, no_repet_n_gramed_data_filename, n):
    print("running create_n_gramed_data.....")
    apis = []
    i=0
    with open(raw_api_filename) as raw_data:
        for line in raw_data:
            line_n_gram = generate_ngrams(line, n)
            tmp_no_repet_n_gram = []
            for gram in line_n_gram:
                if gram not in tmp_no_repet_n_gram:
                    tmp_no_repet_n_gram.append(gram)
            line_no_repet_n_gram = " ".join(tmp_no_repet_n_gram)
            apis.append(line_no_repet_n_gram)
            i += 1
            if i%500==0:
                print(i)
    with open(no_repet_n_gramed_data_filename, 'w')as g:
        for api in apis:
            g.write(api)
        g.close()


def create_n_gramed_data_with_repeat(raw_api_filename, with_repet_n_gramed_data_filename, n):
    print("running create_n_gramed_data.....")
    apis = []
    i=0
    with open(raw_api_filename) as raw_data:
        for line in raw_data:
            line_n_gram = generate_ngrams(line, n)
            line_with_repet_n_gram = " ".join(line_n_gram)
            apis.append(line_with_repet_n_gram)
            i += 1
            if i%500==0:
                print(i)
    with open(with_repet_n_gramed_data_filename, 'w')as g:
        for api in apis:
            g.write(api)
        g.close()

def remove_repet(_list, width):
    n = len(_list)
    if n <= 1:
        print(_list)
        return
    list_rm_repet = []
    for i in range(n - 2 * width + 1):
        isRepet = False
        for j in range(width):
            if _list[i + j] == _list[i + j + width]:
                isRepet = True
            else:
                break
        if isRepet == False:
            list_rm_repet.append(_list[i])
    list_rm_repet.extend(_list[-width:])
    return list_rm_repet


def remove_repet_of_raw_data(raw_api_filename, no_repet_data_filename, long_width):
    print("running remove_repet_of_raw_data ...")
    with open(raw_api_filename) as raw_data:
        apis = []
        count = 0
        for line in raw_data:
            line_list = line.replace('\n', '').split(' ')
            n = len(line_list)
            # if n <= 0:
            #     print(line_list)
            #     continue
            count += 1
            if count % 100 ==0:
                print(count)
            for i in range(long_width):
                # if n <= GLVAR.pic_pow_size * GLVAR.pic_pow_size:
                #     break
                line_list = remove_repet(line_list, i + 1)
                n = len(line_list)

            if n > GLVAR.pic_pow_size*GLVAR.pic_pow_size:
                for i in range(long_width):
                    # if n <= GLVAR.pic_pow_size * GLVAR.pic_pow_size:
                    #     break
                    line_list = remove_repet(line_list, i + 1)
                    # n = len(line_list)
            apis.append(" ".join(line_list))
    apis_str = "\n".join(apis)
    with open(no_repet_data_filename, 'w')as g:
        g.write(apis_str)
        g.close()


def create_n_gram_oprations_set(raw_filename, n_grams_with_repeat_index_filename):
    print("running create_oprations_set ...")
    with open(raw_filename) as raw_data:
        operations = set()
        for line in raw_data:
            operation = line.replace('\n', '').split(' ')
            operations = operations.union(set(operation))

    operations_list = list(operations)
    map_index = range(len(operations))
    ziped_op_index = zip(operations_list, map_index)

    operations_dic = {k: v  for k, v in ziped_op_index}

    with open(n_grams_with_repeat_index_filename, 'w') as json_file:
        json.dump(operations_dic, json_file, ensure_ascii=False)
    print("operations index dictionary create success! Dic file saved in ", n_grams_with_repeat_index_filename)
    print("the operations's count is:", len(operations))


def raw_labels_to_index(raw_lable_filename, lables_index_filename):
    raw_lables_list = []

    with open(raw_lable_filename) as raw_data:
        for line in raw_data:
            raw_lables_list.append(line.replace('\n', '').strip())

    lables_set_index = {}
    lables_index = ""
    for lable in raw_lables_list:
        if lable not in lables_set_index:
            lables_set_index[lable] = len(lables_set_index)
            lables_index += lable + "-" + str(lables_set_index[lable]) + "\n"
            print(lable, "--", lables_set_index[lable])
    print("the lables's count is:", len(lables_set_index))
    lables_index_np = np.zeros(len(raw_lables_list))
    for i, lable in enumerate(raw_lables_list):
        lables_index_np[i] = lables_set_index.get(lable)

    with open(lables_index_filename, "w") as f:
        f.write(lables_index)

    return lables_index_np

def load_npz_data(file_path):
    f = np.load(file_path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def calc_infomation_gain_creat_grams_set(n_gram_data_with_repeat_filename,n_grams_with_repeat_index_filename,n_grams_index_filename):
    print("running calc_infomation_gain ...")
    np.set_printoptions(threshold=np.inf)

    with open(n_grams_with_repeat_index_filename, 'r') as fileR:
        operation_dic = json.load(fileR)
        fileR.close()

    with open(n_gram_data_with_repeat_filename) as raw_data:
        line_num = len(raw_data.readlines())

    with open(n_gram_data_with_repeat_filename) as raw_data:
        length = len(operation_dic)
        print("the total operations's ocunt is:", len(operation_dic))
        processed_data_np = np.empty(shape=(line_num, length)).astype("int32")
        labels_list = []
        for i, line in enumerate(raw_data):
            tmp_processed_data = [0 for x in range(0, length)]
            operation = line.replace('\n', '').split(' ')
            operation_set = set(operation)
            for op in operation_set:
                if len(op) != 0:
                    index = operation_dic[op]
                    tmp_processed_data[index] = 1
            processed_data_np[i] = np.array(tmp_processed_data)

        labels_index_np = raw_labels_to_index(GLVAR.RAW_LABLE_FILENAME,GLVAR.LABLE_INDEX_FINAME)

        x = processed_data_np
        y = labels_index_np

        ig = InformationGain(x, y)
        info_gain = ig.get_result()
        desc_deepcopy_info_gain = copy.deepcopy(info_gain)
        desc_deepcopy_info_gain.sort(reverse = True)

        new_reverse_dic = {v:k for k,v in operation_dic.items()}
        filter_threshold = 0
        if GLVAR.TH_TYPE == 'INFO_GAIN_TH': # 'INFO_GAIN_TH'  or 'SELECT_GRAMS_COUNT'
            print("Threshold use 'INFO_GAIN_TH' , the information gain threshold is:",GLVAR.INFO_GAIN_TH)
            filter_threshold = GLVAR.INFO_GAIN_TH
        else:
            filter_threshold = desc_deepcopy_info_gain[GLVAR.SELECT_GRAMS_COUNT - 1]
            print("Threshold use 'SELECT_GRAMS_COUNT' , the information gain threshold is:",filter_threshold)

        n_gram_set = set()
        i=0
        for ga in info_gain:
            if ga >= filter_threshold:
                n_gram_set.add(new_reverse_dic[i])
            i += 1
        n_grams_list = list(n_gram_set)
        map_index = range(len(n_grams_list))
        ziped_grams_index = zip(n_grams_list,map_index)
        n_grams_dic = {k: v + 1 for k, v in ziped_grams_index}  # V+1 for no operation index is 0

        with open(n_grams_index_filename, 'w') as json_file:
            json.dump(n_grams_dic, json_file, ensure_ascii=False)
        print("filtered n grams index dictionary create success! Dic file saved in: ", GLVAR.N_GRAMS_INDEX_FILENAME)
        print("the grams's count is:", len(n_grams_list))


def process_n_grams_data_4_attention(n_gram_filename, raw_lable_filename, filtered_n_gram_index_filename,
                                 attention_train_data):
    print("running process_raw_data_4_attention ...")
    np.set_printoptions(threshold=np.inf)

    with open(filtered_n_gram_index_filename, 'r') as fileR:
        operation_dic = json.load(fileR)
        fileR.close()
        print("the total operations count is:", len(operation_dic))

    with open(n_gram_filename) as raw_data:
        line_num = len(raw_data.readlines())
        print("raw data total data count is:", line_num)

    with open(n_gram_filename) as raw_data:
        longest_operation_size = 0
        tmp_i = 0
        for i, line in enumerate(raw_data):
            operation = str(line).split(' ')
            selected_operation = [op for op in operation if op in operation_dic.keys()]
            tmp_len = len(selected_operation)
            if tmp_len > longest_operation_size:
                longest_operation_size = tmp_len
                tmp_i = i
        print("longest data RAW operation length is:", longest_operation_size)
        print("longest data index is:",tmp_i)
        longest_operation_size = pow(math.ceil(math.sqrt(longest_operation_size)), 2)
        print("longest data operation length is:", longest_operation_size)
        print("picture size is: ", math.sqrt(longest_operation_size), " * ",
              math.sqrt(longest_operation_size))
    with open(n_gram_filename) as raw_data:
        print("the total operations's ocunt is:", len(operation_dic),
              "\n the longest operation length is",
              longest_operation_size)
        processed_data_np = np.empty(shape=(line_num, longest_operation_size)).astype("int32")
        count = 0
        for i, line in enumerate(raw_data):
            count += 1
            if count % 1000 == 0:
                print(count)
            tmp_processed_data = [0 for x in range(0, longest_operation_size)]
            operation = line.replace('\n', '').split(' ')
            j = 0
            for op in operation:
                if len(op) != 0:
                    index = operation_dic[op]
                    tmp_processed_data[j] = index
                    j += 1
            processed_data_np[i] = np.array(tmp_processed_data)
        labels_index_np = raw_labels_to_index(raw_lable_filename)

        x_train, x_test, y_train, y_test = train_test_split(processed_data_np, labels_index_np,
                                                            test_size=0.2,
                                                            random_state=0)

        np.savez(attention_train_data, x_train=x_train, x_test=x_test, y_train=y_train,
                 y_test=y_test)

def main():

    n=2


    # create_n_gramed_data_with_repeat(GLVAR.RAW_API_FILENAME, GLVAR.N_GRAM_DATA_WITH_REPEAT_FINAME, n)
    #
    #
    # create_n_gram_oprations_set(GLVAR.N_GRAM_DATA_WITH_REPEAT_FINAME,GLVAR.N_GRAMS_WITH_REPEAT_INDEX_FILENAME,)

    calc_infomation_gain_creat_grams_set(GLVAR.N_GRAM_DATA_WITH_REPEAT_FINAME, GLVAR.N_GRAMS_WITH_REPEAT_INDEX_FILENAME,GLVAR.N_GRAMS_INDEX_FILENAME)

    process_n_grams_data_4_attention(GLVAR.N_GRAM_DATA_WITH_REPEAT_FINAME, GLVAR.RAW_LABLE_FILENAME, GLVAR.N_GRAMS_INDEX_FILENAME,
                                 GLVAR.TRAIN_AND_TEST_DATA)

if __name__ == "__main__":
    main()
