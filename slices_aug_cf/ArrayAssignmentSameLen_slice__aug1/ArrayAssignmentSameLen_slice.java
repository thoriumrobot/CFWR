/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        if (false || false) {
            while (true) {
            double __cfwr_var99 = -89.11;
           
        for (int __cfwr_i67 = 0; __cfwr_i67 < 7; __cfwr_i67++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 9; __cfwr_i61++) {
            if ((-58 - null) || ('m' * 37L)) {
            return null;
        }
        }
        }
 break; // Prevent infinite loops
        }
        }

    i_array = array;
    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
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
    int[] c1 = a;
    // See useTest3 for an example of why this assignment should fail.
    @LTLengthOf(
        value = {"c1", "c1"},
        offset = {"0", "x"})
    // :: error: (assignment)
    int z = i;
  }

    protected boolean __cfwr_compute6() {
        for (int __cfwr_i39 = 0; __cfwr_i39 < 8; __cfwr_i39++) {
            if (true || ((null << 63.77) >> 20.72f)) {
            return null;
        }
        }
        return 982;
        return null;
        while (false) {
            for (int __cfwr_i81 = 0; __cfwr_i81 < 8; __cfwr_i81++) {
            short __cfwr_var8 = null;
        }
            break; // Prevent infinite loops
        }
        return false;
    }
    private static long __cfwr_helper912() {
        for (int __cfwr_i37 = 0; __cfwr_i37 < 2; __cfwr_i37++) {
            while (((-331 >> 65.16) - 38.90f)) {
            Character __cfwr_node51 = null;
            break; // Prevent infinite loops
        }
        }
        return (950 << 254L);
        try {
            int __cfwr_elem36 = -744;
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        return 448L;
    }
    Long __cfwr_func326() {
        try {
            if ((('0' << null) / 68.28f) || (-64.25f - 563L)) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 10; __cfwr_i9++) {
            return null;
        }
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        for (int __cfwr_i17 = 0; __cfwr_i17 < 5; __cfwr_i17++) {
            return null;
        }
        return null;
        return null;
    }
}