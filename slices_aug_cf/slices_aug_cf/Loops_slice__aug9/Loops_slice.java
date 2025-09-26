/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        double __cfwr_node53 = ((null | -16.99) - 'y');

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
      private static boolean __cfwr_proc624() {
        try {
            Character __cfwr_val81 = null;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return 'u';
        try {
            try {
            try {
            for (int __cfwr_i57 = 0; __cfwr_i57 < 9; __cfwr_i57++) {
            return null;
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        Long __cfwr_node49 = null;
        return true;
    }
    public static byte __cfwr_util962() {
        for (int __cfwr_i91 = 0; __cfwr_i91 < 5; __cfwr_i91++) {
            while (false) {
            while (true) {
            try {
            if (((-506 * 6.67) | (45.22 >> false)) || false) {
            while (((null | 62.76) >> -18.59f)) {
            return ('Y' << (true + 62.72f));
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        return 'u';
        try {
            try {
            try {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 8; __cfwr_i52++) {
            return (995 ^ -141);
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        Long __cfwr_result1 = null;
        return null;
    }
    char __cfwr_compute956(float __cfwr_p0, float __cfwr_p1, char __cfwr_p2) {
        try {
            if (((10.87 >> 'V') * null) || false) {
            while (false) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 7; __cfwr_i6++) {
            try {
            Character __cfwr_result4 = null;
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        try {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 6; __cfwr_i95++) {
            if (false || false) {
            String __cfwr_data98 = "world60";
        }
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        return 208L;
        short __cfwr_var16 = null;
        return 'S';
    }
}
