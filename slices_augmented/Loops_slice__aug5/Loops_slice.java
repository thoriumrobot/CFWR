/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        char __cfwr_var80 = 'z';
        if (false || (null / false)) {
            return (('U' - 320) * null);
        }


    @Positive
    while (flag) {
      // :: error: (unary.increment)
    @Positive
      offset++;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test1b(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
      // :: error: (compound.assignment)
    @Positive
      offset += 1;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test1c(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
      // :: error: (compound.assignment)
    @Positive
      offset2 += offset;
    @Positive
    }
    @Positive
  }

    @Positive
  public void test2(int[] a, int[] array) {
    @Positive
    int offset = array.length - 1;
    @Positive
    int offset2 = array.length - 1;

    @Positive
    while (flag) {
    @Positive
      offset++;
    @Positive
      offset2 += offset;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int x = offset;
    // :: error: (assignment)
    @Positive
    @LTLengthOf("array") int y = offset2;
    @Positive
  }

    @Positive
  public void test3(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    @Positive
    while (flag) {
    @Positive
      offset--;
      // :: error: (compound.assignment)
    @Positive
      offset2 -= offset;
    @Positive
    }
    @Positive
  }

    protected static int __cfwr_calc465(Long __cfwr_p0) {
        if (false || true) {
            try {
            while ((null - ('s' + 'T'))) {
            if (true && false) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 6; __cfwr_i48++) {
            if (false && false) {
            while (((-451L << -12.32f) / null)) {
            boolean __cfwr_temp25 = false;
            break; // Prevent infinite loops
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        }
        return 700;
    }
    protected long __cfwr_process635(byte __cfwr_p0, Boolean __cfwr_p1) {
        if ((true * (null % null)) || ((false + -44.81f) << null)) {
            byte __cfwr_data12 = null;
        }
        if (true || false) {
            String __cfwr_obj1 = "world22";
        }
        return null;
        try {
            while (true) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 4; __cfwr_i48++) {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 1; __cfwr_i77++) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 6; __cfwr_i83++) {
            return (null % 59L);
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        return 389L;
    }
}