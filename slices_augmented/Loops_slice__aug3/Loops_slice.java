/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
    @Positive
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        String __cfwr_obj94 = "t
        if (true || true) {
            while (true) {
            try {
            if ((-281 & null) && false) {
            while (true) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 1; __cfwr_i75++) {
            try {
            return -432L;
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
est90";

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

    public static Integer __cfwr_proc26(Character __cfwr_p0, boolean __cfwr_p1, Double __cfwr_p2) {
        return null;
        return null;
        return null;
    }
    public byte __cfwr_aux919(Character __cfwr_p0, boolean __cfwr_p1) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 1; __cfwr_i51++) {
            return null;
        }
        if (('8' + (-78.00f & false)) || true) {
            return null;
        }
        Float __cfwr_data68 = null;
        return ((49.64f | true) - null);
    }
}