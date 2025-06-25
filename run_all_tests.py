from tests.test_grading import test_grading_model
from tests.test_segmentation import test_segmentation_model
from tests.test_localization import test_localization_model

if __name__ == "__main__":
    print("\n🔍 Running Grading Test")
    test_grading_model()

    print("\n🔍 Running Segmentation Test")
    test_segmentation_model()

    print("\n🔍 Running Localization Test")
    test_localization_model()