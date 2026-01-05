/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        try {
            return null;
        } catch (Exception __cfwr_e94) {
            // ignore
        }

    @Positive
    i_array = array;
    @Positive
    i_index = index;
    @Positive
  }

    @Positive
  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
    @Positive
    int[] array = a;
    @Positive
        value = {"array", "b"},
    @Positive
        offset = {"0", "-3"})
    // :: error: (assignment)
    @Positive
    int i = index;
    @Positive
  }

    @Positive
  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    @Positive
    int[] c = a;
    // :: error: (assignment)
    @Positive
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @Positive
    @LTLengthOf("c") int y = i;
    @Positive
  }

    private static boolean __cfwr_helper989(int __cfwr_p0) {
        return null;
        while (true) {
            try {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 2; __cfwr_i26++) {
            return (234L * (null / null));
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        try {
            short __cfwr_item54 = null;
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        return false;
    }
    protected static byte __cfwr_aux701(long __cfwr_p0, char __cfwr_p1, String __cfwr_p2) {
        if (true || true) {
            return -68.94f;
        }
        return (-34.00 & null);
    }
}