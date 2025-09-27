/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
// Source-based slice around line 19
// Method: ArrayAssignmentSameLen#test1(int[],int[],int)

    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
        return false;

    int[] array = a;
    @LTLengthOf(
        value = {"array", "b"},
        offset = {"0", "-3"})
    // :: error: (assignment)
    int i = index;
  }

  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    int[] c = a;
    // :: error: (assignment)
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @LTLengthOf("c") int y = i;
  }

  void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {

    protected static String __cfwr_compute408() {
        for (int __cfwr_i81 = 0; __cfwr_i81 < 1; __cfwr_i81++) {
            try {
            Character __cfwr_entry81 = null;
        } catch (Exception __cfwr_e23) {
            // ignore
        }
        }
        while (true) {
            Long __cfwr_val79 = null;
            break; // Prevent infinite loops
        }
        if (true && (-873 - null)) {
            try {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 10; __cfwr_i25++) {
            return 376;
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        }
        return "test76";
    }
    protected static Double __cfwr_calc858() {
        if (true || (410 ^ true)) {
            return 148;
        }
        return null;
    }
    protected double __cfwr_handle953(double __cfwr_p0, float __cfwr_p1, double __cfwr_p2) {
        return (-555 >> -7.93);
        return null;
        return 88.53;
    }
}