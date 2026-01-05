/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        for (int __cfwr_i88 = 0; __cfwr_i88 < 1; __cfwr_i88++) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 10; __cfwr_i44++) {
            try {
            return null;
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
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

    protected static Double __cfwr_handle716(Character __cfwr_p0, Float __cfwr_p1, Boolean __cfwr_p2) {
        while (false) {
            try {
            return null;
        } catch (Exception __cfwr_e99) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    Float __cfwr_process108(byte __cfwr_p0, Double __cfwr_p1) {
        if (true && ((false - false) % (true << -354))) {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 9; __cfwr_i13++) {
            while (('J' << ('x' | 12.43))) {
            Integer __cfwr_temp9 = null;
            break; // Prevent infinite loops
        }
        }
        }
        return null;
        while (true) {
            byte __cfwr_temp31 = ((-661 / false) + -959L);
            break; // Prevent infinite loops
        }
        boolean __cfwr_val18 = true;
        return null;
    }
}