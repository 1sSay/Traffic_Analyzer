class Bbox:
    def __init__(self, label, frame_id, x1, y1, x2, y2, feature_vector=list(), on_border=False, confidence=1):
        self.label = label
        self.frame_id = frame_id
        self.id = -1

        self.x_min = x1
        self.y_min = y1
        self.x_max = x2
        self.y_max = y2

        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.area = self.width * self.height

        self.centre = (self.x_min + self.width // 2, self.y_min + self.height // 2)
        self.prev_centre = (0, 0)
        self.velocity = 0

        self.feature_vector = feature_vector

        self.on_border = on_border
        self.confidence = confidence

    def __repr__(self):
        return self.label, self.frame_id, self.confidence

    def __str__(self):
        return f'{self.label} {self.frame_id} / {self.x_min}, {self.y_min}, {self.x_max}, {self.y_max}'

    def from_fullhd_to_4k(self):
        return self.label, self.frame_id, self.x_min * 2, self.y_min * 2, self.x_max * 2, \
               self.y_max * 2, self.feature_vector, self.on_border, self.confidence

    def get_label(self):
        return self.label

    def get_confidence(self):
        return self.confidence

    def get_frame(self):
        return self.frame_id

    def get_bbox(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def get_bbox_for_CV2rectangle(self):
        return self.x_min, self.y_min, self.width, self.height

    def get_bbox_for_YOLO(self, img_width=3840, img_height=2160):
        x = round((self.x_min + self.width / 2) / img_width, 6)
        y = round((self.y_min + self.height / 2) / img_height, 6)
        w, h = round(self.width / img_width, 6), round(self.height / img_height, 6)
        return f"{self.label} {x} {y} {w} {h}\n"

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_area(self):
        return self.area

    def get_intersection_area(self, second):
        dx = min(self.x_max, second.x_max) - max(self.x_min, second.x_min)
        dy = min(self.y_max, second.y_max) - max(self.y_min, second.y_min)

        if dx > 0 and dy > 0:
            return dx * dy

        return 0

    def get_union_area(self, second):
        return self.area + second.area - self.get_intersection_area(second)

    def get_IoU(self, second):
        intersection_area = self.get_intersection_area(second)

        return intersection_area / (self.area + second.area - intersection_area)

    def get_IoA(self, second):
        intersection_area = self.get_intersection_area(second)

        return intersection_area / self.area, intersection_area / second.area

    def get_centre(self):
        return self.centre

    def get_prev_centre(self):
        return self.prev_centre

    def get_2nd_point_for_arrow(self, arrow_length):
        k = arrow_length / ((abs(self.centre[0] - self.prev_centre[0]) ** 2 + abs(
            self.centre[1] - self.prev_centre[1]) ** 2) ** 0.5)

        return (round(self.prev_centre[0] - (self.prev_centre[0] - self.centre[0]) * k),
                round(self.prev_centre[1] - (self.prev_centre[1] - self.centre[1]) * k))

    def get_1st_point_for_point(self, arrow_length):
        k = arrow_length / ((abs(self.centre[0] - self.prev_centre[0]) ** 2 + abs(
            self.centre[1] - self.prev_centre[1]) ** 2) ** 0.5)

        return (round(self.prev_centre[0] + (self.prev_centre[0] - self.centre[0]) * k),
                round(self.prev_centre[1] + (self.prev_centre[1] - self.centre[1]) * k))
