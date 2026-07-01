############################# IMPORT NECESSARY LIBRARIES #########################################
import math
import os
import zipfile
import numpy as np
import logging
import pickle
import time

from PIL import Image

############################# EVALUATE PERFORMANCE #########################################


def evaluate_performance(model, x_matrix: np.ndarray, y_vector: np.ndarray) -> dict:
    n_samples = len(y_vector)
    predictions = np.zeros(n_samples)

    for i in range(n_samples):
        predictions[i] = model.classify(x_matrix[i])

    tp = np.sum((predictions == 1) & (y_vector == 1))
    tn = np.sum((predictions == -1) & (y_vector == -1))
    fp = np.sum((predictions == 1) & (y_vector == -1))
    fn = np.sum((predictions == -1) & (y_vector == 1))

    condition_positive = tp + fn
    condition_negative = fp + tn

    accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
    tpr = tp / condition_positive if condition_positive > 0 else 0.0
    fpr = fp / condition_negative if condition_negative > 0 else 0.0

    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


############################# INTEGRAL IMAGE #########################################
def integral_image(img_array: np.ndarray) -> np.ndarray:
    prefix_sums = np.cumsum(img_array, axis=1)
    return np.cumsum(prefix_sums, axis=0)


def rectangle_subsum(
    int_img: np.ndarray, i: int, j: int, width: int, height: int
) -> int:
    return (
        int_img[i + width, j + height]
        - int_img[i + width, j]
        - int_img[i, j + height]
        + int_img[i, j]
    )


############################# FEATURES #########################################
class Feature:
    def __init__(
        self, feature_type: str, i: int, j: int, width: int, height: int, value
    ):
        self.feature_type = feature_type
        self.i = i
        self.j = j
        self.width = width
        self.height = height
        self.value = value

    def get(self):
        return self.value

    def show(self):
        print(self)

    def __str__(self) -> str:
        return f"Feature: ({self.feature_type}, {self.i}, {self.j}, {self.width}, {self.height}, {self.value})"

    __repr__ = __str__


def calculate_features(img_array: np.ndarray):
    """Calculate Haar-like features of an image"""

    img_shape = img_array.shape[:2]
    int_img = integral_image(img_array)

    features = []

    # Type a
    for i, j in zip(range(img_shape[0]), range(img_shape[1])):
        for width, height in zip(
            (range((img_shape[0] - i) // 2)), range(img_shape[1] - j)
        ):
            sum1 = rectangle_subsum(int_img, i, j, width, height)
            sum2 = rectangle_subsum(int_img, i + width, j, width, height)
            feat = Feature("a", i, j, width, height, sum1 - sum2)
            features.append(feat)

    # Type b
    for i, j in zip(range(img_shape[0]), range(img_shape[1])):
        for width, height in zip(
            (range((img_shape[0] - i) // 3)), range(img_shape[1] - j)
        ):
            sum1 = rectangle_subsum(int_img, i, j, width, height)
            sum2 = rectangle_subsum(int_img, i + width, j, width, height)
            sum3 = rectangle_subsum(int_img, i + width * 2, j, width, height)
            feat = Feature("b", i, j, width, height, sum1 - sum2 + sum3)
            features.append(feat)

    # Type c
    for i, j in zip(range(img_shape[0]), range(img_shape[1])):
        for width, height in zip(
            (range(img_shape[0] - i)), range((img_shape[1] - j) // 2)
        ):
            sum1 = rectangle_subsum(int_img, i, j, width, height)
            sum2 = rectangle_subsum(int_img, i, j + height, width, height)
            feat = Feature("c", i, j, width, height, sum1 - sum2)
            features.append(feat)

    # Type d
    for i, j in zip(range(img_shape[0]), range(img_shape[1])):
        for width, height in zip(
            (range(img_shape[0] - i)), range((img_shape[1] - j) // 3)
        ):
            sum1 = rectangle_subsum(int_img, i, j, width, height)
            sum2 = rectangle_subsum(int_img, i, j + height, width, height)
            sum3 = rectangle_subsum(int_img, i, j + height * 2, width, height)
            feat = Feature("d", i, j, width, height, sum1 - sum2 + sum3)
            features.append(feat)

    # Type e
    for i, j in zip(range(img_shape[0]), range(img_shape[1])):
        for width, height in zip(
            (range((img_shape[0] - i) // 2)), range((img_shape[1] - j) // 3)
        ):
            sum1 = rectangle_subsum(int_img, i, j, width, height)
            sum2 = rectangle_subsum(int_img, i, j + height, width, height)
            sum3 = rectangle_subsum(int_img, i, j + height * 2, width, height)
            sum4 = rectangle_subsum(int_img, i + width, j + height, width, height)
            feat = Feature("e", i, j, width, height, sum1 - sum2 - sum3 + sum4)
            features.append(feat)

    return np.array(features)


############################# CLASSIFIER CLASSES #########################################
class WeakClassifier:
    def __init__(
        self,
        threshold: int = 0,
        toggle: int = 0,
        error: int = 0,
        margin: int = 0,
        feature: int = 0,
    ):
        self.threshold = threshold
        self.toggle = toggle
        self.error = error
        self.margin = margin
        self.feature = feature
        pass

    def classify(self, feature) -> int:
        if self.toggle * feature < self.toggle * self.threshold:
            return 1
        else:
            return -1


class StrongClassifier:
    def __init__(
        self,
        weak_classifiers: list[WeakClassifier] | None = None,
        alphas: list[float] | None = None,
    ):
        self.weak_classifiers = [] if weak_classifiers is None else weak_classifiers
        self.alphas = [] if alphas is None else alphas
        self.threshold = 0.0

    def compute_margin(self, feature_vec: np.ndarray) -> float:
        score = 0.0
        for alpha, weak in zip(self.alphas, self.weak_classifiers):
            feat_val = feature_vec[weak.feature]
            score += alpha * weak.classify(feat_val)
        return score

    def append_classifier(self, classifier: WeakClassifier):
        error = classifier.error

        if error != 0:
            bounded_error = np.clip(error, 1e-15, 1.0 - 1e-15)
            alpha = 0.5 * math.log((1.0 - bounded_error) / bounded_error)
        else:
            alpha = 1

        self.weak_classifiers.append(classifier)
        self.alphas.append(alpha)

        self.threshold = 0.5 * sum(self.alphas)

    def last_classifier(self):
        return self.weak_classifiers[-1]

    def classify(self, feature_vec) -> int:
        score = 0.0

        for alpha, weak in zip(self.alphas, self.weak_classifiers):
            feat_val = feature_vec[weak.feature]
            score += alpha * weak.classify(feat_val)

        return 1 if score >= self.threshold else -1

    def adjust_thresold(self, features_matrix, labels, target):
        positive_indices = np.where(labels == 1)[0]
        positive_features = features_matrix[positive_indices]

        scores = np.zeros(len(positive_features))
        for i, feat_vec in enumerate(positive_features):
            for alpha, weak in zip(self.alphas, self.weak_classifiers):
                scores[i] += alpha * weak.classify(feat_vec[weak.feature])

        scores = np.sort(scores)

        margin_index = int(np.floor(len(scores) * (1.0 - target)))

        if margin_index < len(scores):
            self.threshold = scores[margin_index]
        else:
            self.threshold = scores[-1]


############################# ADABOOST FUNCTIONS #########################################


def decision_stump(training_projections, weights, feature: int) -> WeakClassifier:
    """training projections list must be sorted in ascending order"""

    threshold = training_projections[0, 0] - 1
    toggle = 0
    margin = 0
    error = 2

    x = training_projections[:, 0]
    y = training_projections[:, 1]

    original_indices = training_projections[:, 2].astype(int)
    aligned_weights = weights[original_indices]

    n = len(y)

    # feature is larger than the present threshold
    w_pos_plus = np.sum(aligned_weights[y == 1])
    w_neg_plus = np.sum(aligned_weights[y == -1])

    # feature is less than the present threshold
    w_pos_min = 0
    w_neg_min = 0

    j = -1
    temp_threshold = threshold
    temp_margin = margin

    while 1:
        # select the toggle to minimize the weighted error
        error_plus_one = w_neg_plus + w_pos_min
        error_minus_one = w_pos_plus + w_neg_min

        # change temp toggle 1,-1
        if error_plus_one < error_minus_one:
            temp_error = error_plus_one
            temp_toggle = -1
        else:
            temp_error = error_minus_one
            temp_toggle = 1

        if temp_error < error or (temp_error == error and temp_margin > margin):
            error = temp_error
            margin = temp_margin
            threshold = temp_threshold
            toggle = temp_toggle

        if j == n - 1:
            break

        j += 1

        while 1:
            if y[j] == -1:
                w_neg_min += aligned_weights[j]
                w_neg_plus -= aligned_weights[j]

            else:
                w_pos_min += aligned_weights[j]
                w_pos_plus -= aligned_weights[j]

            if j == n - 1 or x[j] != x[j + 1]:
                # no new valid threshold
                break
            else:
                j += 1

        if j == n - 1:
            temp_threshold = x[n - 1] + 1
            temp_margin = 0
        else:
            temp_threshold = (x[j] + x[j + 1]) / 2
            temp_margin = x[j + 1] - x[j]

    return WeakClassifier(
        threshold=threshold, toggle=toggle, error=error, margin=margin, feature=feature
    )


def best_stump(training_examples, active_features: list[bool], weights):
    d = len(active_features)
    best_classifier = WeakClassifier(
        error=2,
        margin=0,
    )

    for feat_num in range(d):
        if not active_features[feat_num]:
            continue

        classifier = decision_stump(training_examples[feat_num], weights, feat_num)
        if classifier.error < best_classifier.error or (
            classifier.error == best_classifier.error
            and classifier.margin > best_classifier.margin
        ):
            best_classifier = classifier

    return best_classifier


def initialize_empty_weights(training_examples):
    # TODO: consider basic initialization 1/n
    y = training_examples[0][:, 1]

    positives = np.count_nonzero(y == 1)
    negatives = len(y) - positives

    weights = np.full(y.shape, 1 / (2 * negatives))
    weights[y == 1] = 1 / (2 * positives)

    return weights


def initialize_weights(
    training_projection, previous_classifier: WeakClassifier, previous_weights
):
    error = previous_classifier.error

    x = training_projection[:, 0]
    y = training_projection[:, 1]
    n = len(x)

    if error == 0:
        return initialize_empty_weights([training_projection])

    pred = np.array([previous_classifier.classify(e) for e in x])
    is_error = pred != y

    updated_weights = np.where(
        is_error, previous_weights / (2 * error), previous_weights / (2 * (1 - error))
    )

    return updated_weights / np.sum(updated_weights)


def adaboost(
    training_examples,
    training_rounds: int,
    active_features,
    previous_weights=None,
    classifier: StrongClassifier | None = None,
) -> tuple[StrongClassifier, np.ndarray]:

    if classifier is None or previous_weights is None:
        weights = initialize_empty_weights(training_examples)
        classifier = StrongClassifier()
    else:
        last_classifier = classifier.last_classifier()
        last_feat = last_classifier.feature
        weights = initialize_weights(
            training_examples[last_feat], last_classifier, previous_weights
        )
        weights = previous_weights

    for t in range(training_rounds):
        weak_classifier = best_stump(training_examples, active_features, weights)
        classifier_feat = weak_classifier.feature
        active_features[classifier_feat] = False

        classifier.append_classifier(weak_classifier)

        if t == 0 and weak_classifier.error == 0:
            return classifier, weights
        else:
            weights = initialize_weights(
                training_examples[classifier_feat], weak_classifier, weights
            )

    return classifier, weights


############################# ATTENTIONAL CASCADE #########################################
class AttentionalCascade:
    def __init__(self):
        self.stages: list[StrongClassifier] = []

    def add_stage(self, stage: StrongClassifier | None):
        if stage is None:
            pass
        else:
            self.stages.append(stage)

    def classify(self, feature_vector) -> int:
        """If any of the classifiers returns -1, the sub-window if rejected"""
        for stage in self.stages:
            if stage.classify(feature_vector) == -1:
                return -1
        return 1


def generate_perlin_noise_samples(n: int, shape: tuple = (24, 24), res: tuple = (4, 4)):
    images = []

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    def f(t):
        return t**3 * (t * (t * 6 - 15) + 10)

    for _ in range(n):
        theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.stack((np.cos(theta), np.sin(theta)), axis=-1)

        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

        n00 = np.sum(grid * g00, axis=-1)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=-1)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=-1)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=-1)

        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + n10 * t[:, :, 0]
        n1 = n01 * (1 - t[:, :, 0]) + n11 * t[:, :, 0]
        noise = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

        min_val = np.min(noise)
        max_val = np.max(noise)
        scaled = (noise - min_val) / (max_val - min_val + 1e-9)

        images.append(scaled)

    return images


def mine_real_negatives(image_path: str, n: int, shape: tuple = (24, 24)):
    other_x = np.load(image_path)["X"]
    np.random.shuffle(other_x)

    return other_x[: n]


def generate_noise_samples(n: int, shape: tuple = (24, 24)):
    images = []
    for _ in range(n):
        noise = np.random.normal(loc=128, scale=50, size=shape)
        noise = np.clip(noise, 0, 255).astype(np.uint8)

        mean = np.mean(noise)
        var = np.var(noise)
        eps = 1e-9

        normalized = (noise - mean) / (np.sqrt(var) + eps)

        min_val = np.min(normalized)
        max_val = np.max(normalized)
        scaled = (normalized - min_val) / (max_val - min_val + 1e-9)

        images.append(scaled)

    return images


def generate_negative_samples(n: int, shape: tuple = (24, 24), mode : str = "normal"):
    # logging.info(f"generate_{mode}: {n}")

    if mode == "normal":
        return generate_noise_samples(n, shape)
    elif mode == "perlin":
        return generate_perlin_noise_samples(n, shape)
    else:
        return mine_real_negatives("negative_examples.npz", n, shape)


def train_attentional_cascade(
    positive_features,
    negative_features,
    active_features,
    feature_templates,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    fp_stage_target: float = 0.5,
    detection_target: float = 0.995,
    fp_cascade_target: float = 1e-5,
    max_stage: int = 15,
) -> tuple[AttentionalCascade, dict]:

    training_data = {
        "stage_metrics": [],
        "test_metrics": [],
        "feature_geometries": [],
        "margins_positive": [],
        "margins_negative": [],
    }

    cascade = AttentionalCascade()
    fp_cascade_curr = 1.0
    detection_curr = 1.0
    stage = 0

    positive_labels = np.ones(len(positive_features))
    negative_labels = -np.ones(len(negative_features))

    while fp_cascade_curr > fp_cascade_target and stage < max_stage:
        stage += 1
        stage_start_time = time.time()
        logging.info(f" Training stage {stage}")

        stage_active_features = active_features.copy()

        x = np.vstack((positive_features, negative_features))
        y = np.concatenate((positive_labels, negative_labels))

        n_features = x.shape[1]
        train_x = []

        for idx in range(n_features):
            feat_projection = np.column_stack((x[:, idx], y, np.arange(len(y))))
            feat_projection = feat_projection[feat_projection[:, 0].argsort()]

            train_x.append(feat_projection)

        stage_classifier = None
        stage_weights = None
        fp_i = 1

        while fp_i > fp_stage_target:
            stage_classifier, stage_weights = adaboost(
                training_examples=train_x,
                training_rounds=1,
                previous_weights=stage_weights,  # changed
                active_features=stage_active_features,
                classifier=stage_classifier,
            )

            stage_classifier.adjust_thresold(x, y, detection_target)

            false_positives = 0
            for negative_feature in negative_features:
                if stage_classifier.classify(negative_feature) == 1:
                    false_positives += 1
            fp_i = false_positives / len(negative_features)
    
            max_weak_classifiers = 10 + (stage * 20)
            
            if fp_i == 0 or len(stage_classifier.weak_classifiers) >= max_weak_classifiers:
                if len(stage_classifier.weak_classifiers) >= max_weak_classifiers:
                    logging.warning(f"Stage {stage} hit classifier limit ({max_weak_classifiers}) with FPR: {fp_i:.4f}")
                break

        cascade.add_stage(stage_classifier)
        fp_cascade_curr *= fp_i
        detection_curr *= detection_target

        # Update performance metrics
        stage_duration = time.time() - stage_start_time
        logging.info(
            f"Stage {stage} complete. Cascade false positive rate: {fp_cascade_curr:.6f}, Cascade detection rate: {detection_curr:.4f}"
        )

        assert stage_classifier is not None

        geometries = [
            (
                feature_templates[wc.feature].feature_type,
                feature_templates[wc.feature].i,
                feature_templates[wc.feature].j,
                feature_templates[wc.feature].width,
                feature_templates[wc.feature].height,
                alpha,
            )
            for wc, alpha in zip(
                stage_classifier.weak_classifiers, stage_classifier.alphas
            )
        ]

        pos_margins = np.array(
            [stage_classifier.compute_margin(feat) for feat in positive_features]
        )
        neg_margins = np.array(
            [stage_classifier.compute_margin(feat) for feat in negative_features]
        )

        training_data["stage_metrics"].append(
            {
                "stage": stage,
                "fpr_stage": fp_i,
                "fpr_cumulative": fp_cascade_curr,
                "tpr_cumulative": detection_curr,
                "time_seconds": stage_duration,
                "n_classifiers": len(stage_classifier.weak_classifiers),
                "negative_pool_size": len(negative_features),
                "stage_threshold": stage_classifier.threshold,
            }
        )
        training_data["feature_geometries"].append(geometries)
        training_data["margins_positive"].append(pos_margins)
        training_data["margins_negative"].append(neg_margins)

        test_stats = evaluate_performance(cascade, test_features, test_labels)
        training_data["test_metrics"].append(
            {
                "stage": stage,
                "accuracy": test_stats["accuracy"],
                "tpr": test_stats["tpr"],
                "fpr": test_stats["fpr"],
            }
        )
        logging.info(
            f"Test Set Evaluation (Stage {stage}) | "
            f"Accuracy: {test_stats['accuracy']:.4f} | "
            f"TPR: {test_stats['tpr']:.4f} | "
            f"FPR: {test_stats['fpr']:.6f}"
        )

        logging.info(f"Cascade current fp: {fp_cascade_curr}, target: {fp_cascade_target}")

        if fp_cascade_curr > fp_cascade_target:
            new_negative_features = []

            for negative_feature in negative_features:
                if cascade.classify(negative_feature) == 1:
                    new_negative_features.append(negative_feature)

            m = len(negative_features) - len(new_negative_features)
            BATCH_SIZE = m
            attempts = 0
            while m > 0:
                attempts += 1
                # logging.info(f"New {m} negative samples")
                if attempts < 5:
                    noise_samples = generate_negative_samples(BATCH_SIZE * 10, mode="normal")
                elif attempts < 10:
                    noise_samples = generate_negative_samples(BATCH_SIZE * 10, mode="perlin")
                else:
                    noise_samples = generate_negative_samples(BATCH_SIZE * 10, mode="real")

                for sample in noise_samples:
                    features = calculate_features(sample)
                    feature_vec = np.array([f.get() for f in features])

                    if cascade.classify(feature_vec) == 1:
                        new_negative_features.append(feature_vec)
                        m -= 1
                    if len(new_negative_features) == len(negative_features):
                        break

            negative_features = np.array(
                new_negative_features[: len(negative_features)]
            )

    # logging.info(f"gathered data: {training_data}")
    return cascade, training_data


############################# PROCESS DIRECTORY #########################################

def process_dir(
    directory: str, prefix: str, label: int, target_shape: tuple[int, int] = (24, 24)
) -> tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []

    if not os.path.exists(directory):
        logging.warning(f"Directory missing, skipping: {directory}")
        return np.array(x_list), np.array(y_list)

    for filename in sorted(os.listdir(directory)):

        filepath = os.path.join(directory, filename)
        with Image.open(filepath) as img:
            img = img.convert("L")
            img = img.resize(target_shape, Image.Resampling.BILINEAR)

            matrix = np.asarray(img, dtype=np.uint8)

            mean = np.mean(matrix)
            var = np.var(matrix)
            eps = 1e-9
            matrix = (matrix - mean) / (np.sqrt(var) + eps)

            x_list.append(matrix)
            y_list.append(label)

    return np.array(x_list), np.array(y_list)


############################# PREPARE AND LOAD DATA #########################################


def prepare_dataset(zip_path: str, extract_to: str = "train_set"):
    if os.path.exists(extract_to):
        logging.warning(f"Skipping extraction. Directory {extract_to} already exists.")
        return

    logging.info(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip:
        zip.extractall(".")
    logging.info(f"Extraction complete.")


def extract_feature_matrix(arr: np.ndarray) -> np.ndarray:
    matrix = []
    for index, image in enumerate(arr):
        features = calculate_features(image)
        feature_vals = [f.get() for f in features]
        matrix.append(feature_vals)
        if index % 100 == 0 or index == (len(arr) - 1):
            logging.info(f"Processed {index + 1} out of {len(arr)} images")

    return np.array(matrix)


def run_pipeline(zip_path: str, test_path: str, output_path: str = "model.pkl"):
    logging.basicConfig(
            filename="train.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s")

    faces_x = np.load("train_set_faces.npz")["X"]
    other_x = np.load("train_set_other.npz")["X"]

    if len(faces_x) == 0 or len(other_x) == 0:
        raise ValueError("Dataset parsing failed. Verify directory paths.")

    logging.info("Extracting feature matrix for Positive Examples (Faces)...")
    positive_feature_matrix = extract_feature_matrix(faces_x)

    logging.info("Extracting feature matrix for Negative Examples (Other)...")
    negative_feature_matrix = extract_feature_matrix(other_x)

    logging.info(f"Extracting test set from {test_path}...")

    test_faces_x = np.load("test_set_faces.npz")["X"]
    test_other_x = np.load("test_set_other.npz")["X"]

    test_pos_features = extract_feature_matrix(test_faces_x)
    test_neg_features = extract_feature_matrix(test_other_x)

    test_feature_matrix = np.vstack((test_pos_features, test_neg_features))
    test_labels = np.concatenate(
        (np.ones(len(test_pos_features)), -np.ones(len(test_neg_features)))
    )

    target_shape = faces_x[0].shape
    feature_templates = calculate_features(np.zeros(target_shape))

    total_features = positive_feature_matrix.shape[1]
    active_features = [True] * total_features
    logging.info(f"Total extracted Haar features per image: {total_features}")

    logging.info("Starting Attentional Cascade Training...")
    cascade, data = train_attentional_cascade(
        positive_features=positive_feature_matrix,
        negative_features=negative_feature_matrix,
        active_features=active_features,
        feature_templates=feature_templates,
        test_features=test_feature_matrix,
        test_labels=test_labels,
        fp_stage_target=0.5,
        detection_target=0.995,
        fp_cascade_target=1e-3,
    )

    artifact = {"model": cascade, "data": data}

    logging.info(f"Saving trained weights to {output_path}")
    with open(output_path, "wb") as output:
        pickle.dump(artifact, output, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Pipeline execution completed successfully.")


############################# TRAIN #########################################
if __name__ == "__main__":
    rand_nb = np.random.randint(21376769)

    logging.basicConfig(
        filename=f"train_{rand_nb}.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    run_pipeline(
        zip_path="dataset.zip",
        test_path="dataset.zip",
        output_path=f"viola_jones_cascade_{rand_nb}.pkl",
    )