/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        for (int __cfwr_i73 = 0; __cfwr_i73 < 8; __cfwr_i73++) {
            try {
            try {
            return -533L;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
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

    static long __cfwr_temp339(float __cfwr_p0) {
        try {
            if (false && false) {
            try {
            long __cfwr_entry42 = -374L;
        } catch (Exception __cfwr_e2) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        for (int __cfwr_i50 = 0; __cfwr_i50 < 2; __cfwr_i50++) {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 8; __cfwr_i78++) {
            return (63 - -345);
        }
        }
        return (-43.39f ^ true);
    }
    private float __cfwr_aux110(Double __cfwr_p0, Character __cfwr_p1) {
        try {
            if (((true << 93.30) / 'c') && false) {
            return -660;
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
        return -87.39f;
    }
}