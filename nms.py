from settings import confidence_threshold, width_and_height_threshold, area_threshold

def ExtendedNonMaximumSupression(objects,
                                 nms_threshold=0.4,
                                 destroy_objects_on_border=False,
                                 destroy_low_confidence_objects=False,
                                 check_IoA=False):

    objects.sort(key=lambda obj: obj.confidence, reverse=True)

    keep_list = list()
    drop_list = set()

    for i in range(len(objects)):
        if i in drop_list:
            continue

        if destroy_objects_on_border and objects[i].on_border:
            objects[i].confidence /= 2

        if destroy_low_confidence_objects and objects[i].get_confidence() < confidence_threshold:
            drop_list.add(i)
            continue

        for j in range(i + 1, len(objects)):
            if j in drop_list:
                continue

            IoU = objects[i].get_IoU(objects[j])
            if IoU > nms_threshold and objects[i].get_confidence() > objects[j].get_confidence():
                drop_list.add(j)
                continue
            if IoU > nms_threshold and objects[i].get_confidence() <= objects[j].get_confidence():
                drop_list.add(i)
                break

            if check_IoA:
                IoA_i, IoA_j = objects[i].get_IoA(objects[j])
                if IoA_j > 0.8 and objects[i].area > objects[j].area:
                    drop_list.add(j)
                    continue
                if IoA_i > 0.8 and objects[i].area < objects[j].area:
                    drop_list.add(i)
                    break

        if i not in drop_list:
            keep_list.append(objects[i])

    return keep_list
