def evaluation(class_dir, threshold = 0.5):
    f = open(class_dir + '/result.txt', 'r')
    ls = f.readlines()
    f.close()

    ne_out = []
    exp_out = []
    for l in ls:
        part = l.split(' ')
        if part[0] == '0.0':
            ne_out.append(float(part[1].replace('\n', '')))
        elif part[0] == '1.0':
            exp_out.append(float(part[1].replace('\n', '')))

    ne_num = 0
    exp_num = 0
    for ne in ne_out:
        if ne < 0.5:
            ne_num += 1
    for exp in exp_out:
        if exp >= 0.5:
            exp_num += 1

    if len(ne_out) == 0:
        ne_ratio = ne_num
    else:
        ne_ratio = ne_num / len(ne_out)
    if len(exp_out) == 0:
        exp_ratio = exp_num
    else:
        exp_ratio = exp_num / len(exp_out)

    print('neutral : ' + str(ne_ratio))
    print('expression : ' + str(exp_ratio))


if __name__ == '__main__':
    evaluation()

