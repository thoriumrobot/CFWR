/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        try {
            byte _
        if (false && false) {
            return (('A' % null) % (-382L & true));
        }
_cfwr_result95 = ('w' << true);
        } catch (Exception __cfwr_e87) {
            // ignore
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

    private byte __cfwr_util53() {
        for (int __cfwr_i71 = 0; __cfwr_i71 < 9; __cfwr_i71++) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 3; __cfwr_i97++) {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 10; __cfwr_i50++) {
            double __cfwr_obj8 = (-60.42f / (null + true));
        }
        }
        }
        return null;
    }
    protected static Float __cfwr_proc137() {
        try {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 10; __cfwr_i17++) {
            while (true) {
            Double __cfwr_entry18 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e61) {
            // ignore
        }
        return null;
    }
}