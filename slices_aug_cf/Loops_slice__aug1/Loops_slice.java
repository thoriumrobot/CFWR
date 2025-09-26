/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Loops_slice {
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        if ((-84.83f ^ ('c' / '6')) || true) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 8; __cfwr_i32++) {
            while (false) {
            if (true && true) {
            return (62L ^ (null | null));
        }
            break; // Prevent infinite loops
        }
        }
        }

    while (flag) {
      // :: error: (unary.increment)
      offset++;
    }
  }

  public void test1b(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (compound.assignment)
      offset += 1;
    }
  }

  public void test1c(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (compound.assignment)
      offset2 += offset;
    }
  }

  public void test2(int[] a, int[] array) {
    int offset = array.length - 1;
    int offset2 = array.length - 1;

    while (flag) {
      offset++;
      offset2 += offset;
    }
    // :: error: (assignment)
    @LTLengthOf("array") int x = offset;
    // :: error: (assignment)
    @LTLengthOf("array") int y = offset2;
  }

  public void test3(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      offset--;
      // :: error: (compound.assignment)
      offset2 -= offset;
    }
  }

  public void test4(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
    while (flag) {
      // :: error: (unary.increment)
      offset++;
      // :: error: (compound.assignment)
      offset += 1;
      // :: error: (compound.assignment)
      offset2 += offset;
    }
  }

  public void test4(int[] src) {
    int patternLength = src.length;
    int[] optoSft = new int[patternLength];
    for (int i = patternLength; i > 0; i--) {}
  }

  public void test5(
      int[] a,
      @LTLengthOf(value = "#1", offset = "-1000") int offset,
      @LTLengthOf("#1") int offset2) {
    int otherOffset = offset;
    while (flag) {
      otherOffset += 1;
      // :: error: (unary.increment)
      offset++;
      // :: error: (compound.assignment)
      offset += 1;
      // :: error: (compound.assignment)
      offset2 += offset;
    }
    // :: error: (assignment)
    @LTLengthOf(value = "#1", offset = "-1000") int x = otherOffset;
  }

    private static short __cfwr_calc788(long __cfwr_p0, boolean __cfwr_p1) {
        while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e93) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        if (true || false) {
            try {
            while (true) {
            for (int __cfwr_i17 = 0; __cfwr_i17 < 1; __cfwr_i17++) {
            while (false) {
            byte __cfwr_temp88 = ((553L * 92.18f) % null);
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
        return null;
    }
    protected char __cfwr_handle489(Object __cfwr_p0, Character __cfwr_p1, char __cfwr_p2) {
        return "world29";
        while (true) {
            double __cfwr_result62 = 97.39;
            break; // Prevent infinite loops
        }
        while (true) {
            while (true) {
            try {
            return null;
        } catch (Exception __cfwr_e5) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
        return 'q';
    }
}