/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayIntro_slice {
    @Positive
  void test() {
        return null;

    @Positive
    int @MinLen(5) [] arr = new int[5];
    @Positive
    int a = 9;
    @Positive
    a += 5;
    @Positive
    a -= 2;
    @Positive
    int @MinLen(12) [] arr1 = new int[a];
    @Positive
    int @MinLen(3) [] arr2 = {1, 2, 3};
    // :: error: (assignment)
    @Positive
    int @MinLen(4) [] arr3 = {4, 5, 6};
    // :: error: (assignment)
    @Positive
    int @MinLen(7) [] arr4 = new int[4];
    // :: error: (assignment)
    @Positive
    int @MinLen(16) [] arr5 = new int[a];
    @Positive
  }

    private static byte __cfwr_aux224(short __cfwr_p0, Double __cfwr_p1) {
        try {
            return null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return ('l' % 'R');
        return (990L | -854L);
    }
}