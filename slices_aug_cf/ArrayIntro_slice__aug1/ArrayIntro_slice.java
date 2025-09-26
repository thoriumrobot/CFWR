/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayIntro_slice {
  void test() {
        float __cfwr_data59 = 42.40f;

    int @MinLen(5) [] arr = new int[5];
    int a = 9;
    a += 5;
    a -= 2;
    int @MinLen(12) [] arr1 = new int[a];
    int @MinLen(3) [] arr2 = {1, 2, 3};
    // :: error: (assignment)
    int @MinLen(4) [] arr3 = {4, 5, 6};
    // :: error: (assignment)
    int @MinLen(7) [] arr4 = new int[4];
    // :: error: (assignment)
    int @MinLen(16) [] arr5 = new int[a];
  }

    public static Boolean __cfwr_func337(Character __cfwr_p0, double __cfwr_p1) {
        return false;
        byte __cfwr_node4 = null;
        return null;
    }
}