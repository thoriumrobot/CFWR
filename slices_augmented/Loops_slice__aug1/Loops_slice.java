/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        try {
            return (-901 | 424);
        } catch (Exception __cfwr_e56) {
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

    Integer __cfwr_helper63(char __cfwr_p0, Long __cfwr_p1, String __cfwr_p2) {
        if (true && (84.03f << -593L)) {
            if (true && false) {
            try {
            return -395L;
        } catch (Exception __cfwr_e48) {
            // ignore
        }
        }
        }
        try {
            return 'H';
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        String __cfwr_var89 = "value40";
        return null;
    }
    protected float __cfwr_helper491(Float __cfwr_p0, long __cfwr_p1, Float __cfwr_p2) {
        while (false) {
            while (true) {
            Double __cfwr_var41 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return -19.54f;
    }
    private Long __cfwr_temp493(short __cfwr_p0, byte __cfwr_p1) {
        if (true || false) {
            return null;
        }
        int __cfwr_val12 = -744;
        for (int __cfwr_i2 = 0; __cfwr_i2 < 7; __cfwr_i2++) {
            while (true) {
            while (('s' ^ null)) {
            if (true || (null / (773L & true))) {
            long __cfwr_entry32 = (null * (44.65 % -51.48));
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
    }
}