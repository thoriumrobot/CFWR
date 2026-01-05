/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        for (int __cfwr_i92 = 0;
        int __cfwr_result56 = -374;
 __cfwr_i92 < 8; __cfwr_i92++) {
            try {
            while (true) {
            for (int __cfwr_i73 = 0; __cfwr_i73 < 5; __cfwr_i73++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e5) {
            // ignore
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

    protected static Long __cfwr_handle69(Float __cfwr_p0, short __cfwr_p1) {
        try {
            try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 9; __cfwr_i57++) {
            while (('1' / (-954 ^ 67.80f))) {
            return null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e71) {
            // ignore
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return null;
    }
    public static Integer __cfwr_temp668(boolean __cfwr_p0) {
        if (false || true) {
            while (true) {
            Long __cfwr_data38 = null;
            break; // Prevent infinite loops
        }
        }
        for (int __cfwr_i67 = 0; __cfwr_i67 < 2; __cfwr_i67++) {
            if ((-26.97 - null) || true) {
            while (false) {
            Double __cfwr_var56 = null;
            break; // Prevent infinite loops
        }
        }
        }
        try {
            try {
            return null;
        } catch (Exception __cfwr_e72) {
            // ignore
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        return "temp33";
        return null;
    }
}