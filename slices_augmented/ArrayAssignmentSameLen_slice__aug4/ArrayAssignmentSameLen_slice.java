/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
    @Positive
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        return null;

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

    Integer __cfwr_util983(long __cfwr_p0, Float __cfwr_p1) {
        try {
            try {
            Long __cfwr_temp96 = null;
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        for (int __cfwr_i54 = 0; __cfwr_i54 < 7; __cfwr_i54++) {
            Float __cfwr_var72 = null;
        }
        for (int __cfwr_i6 = 0; __cfwr_i6 < 10; __cfwr_i6++) {
            if (true || ('S' + '3')) {
            short __cfwr_obj76 = null;
        }
        }
        if (('x' % -173L) && true) {
            if (false || false) {
            try {
            return null;
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        }
        }
        return null;
    }
    byte __cfwr_util586(short __cfwr_p0, long __cfwr_p1, Long __cfwr_p2) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        if (((-424 / 877L) & null) && (-81.90f | 765L)) {
            while (((-645 | null) + 'w')) {
            while (false) {
            while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e78) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
    private static Object __cfwr_helper859() {
        return 'b';
        if (false && false) {
            return null;
        }
        return null;
    }
}