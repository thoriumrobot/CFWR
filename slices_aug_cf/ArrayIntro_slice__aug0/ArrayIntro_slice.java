/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayIntro_slice {
  void test() {
        double __cfwr_var10 = -98.87;

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

    public static double __cfwr_compute672(float __cfwr_p0, short __cfwr_p1, boolean __cfwr_p2) {
        Integer __cfwr_obj5 = null;
        if (false && (null >> null)) {
            return (true << (864 >> null));
        }
        return 77.66;
    }
}