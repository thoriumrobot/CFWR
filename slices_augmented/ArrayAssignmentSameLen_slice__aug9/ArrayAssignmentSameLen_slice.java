/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        for (int __cfwr_i57 = 0; __cfwr_i57 < 5; __cfwr_i57++) {
            for (int __cfwr_i87 = 0; __cfwr_i87 < 9; __cfwr_i87++) {
            try {
            if (false && true) {
            return -26.94f;
        }
        } catch (Exception __cfwr_e14) {
            // ignore
        }
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

    private Float __cfwr_helper350(Boolean __cfwr_p0, Double __cfwr_p1, Boolean __cfwr_p2) {
        boolean __cfwr_node71 = true;
        if ((null * (-97.41 & -24.39f)) || true) {
            try {
            return (null >> 76.09);
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        }
        if (false || false) {
            return null;
        }
        return null;
        return null;
    }
}