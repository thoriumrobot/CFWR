/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        if (true || (null ^ 592)
        return 560;
) {
            while (false) {
            short __cfwr_entry30 = ((null ^ 'f') | null);
            break; // Prevent infinite loops
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

    public static String __cfwr_func903(Character __cfwr_p0) {
        return (996 | (null % false));
        return null;
        return "temp22";
    }
    protected static char __cfwr_calc330(byte __cfwr_p0, Object __cfwr_p1) {
        while (true) {
            if ((null << (-73.53 * null)) || true) {
            short __cfwr_elem33 = null;
        }
            break; // Prevent infinite loops
        }
        if (('T' / (null % 48.82)) || false) {
            while (true) {
            try {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 7; __cfwr_i1++) {
            if (false && true) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 3; __cfwr_i28++) {
            if (false || false) {
            Character __cfwr_entry93 = null;
        }
        }
        }
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        try {
            while (true) {
            if (true && false) {
            String __cfwr_item72 = "hello21";
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        while (false) {
            for (int __cfwr_i63 = 0; __cfwr_i63 < 1; __cfwr_i63++) {
            try {
            Character __cfwr_result90 = null;
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        return 's';
    }
    private Long __cfwr_helper926(Boolean __cfwr_p0) {
        for (int __cfwr_i59 = 0; __cfwr_i59 < 6; __cfwr_i59++) {
            for (int __cfwr_i68 = 0; __cfwr_i68 < 9; __cfwr_i68++) {
            try {
            if (false && true) {
            return null;
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
        }
        }
        for (int __cfwr_i7 = 0; __cfwr_i7 < 8; __cfwr_i7++) {
            for (int __cfwr_i27 = 0; __cfwr_i27 < 7; __cfwr_i27++) {
            Integer __cfwr_node38 = null;
        }
        }
        return null;
    }
}