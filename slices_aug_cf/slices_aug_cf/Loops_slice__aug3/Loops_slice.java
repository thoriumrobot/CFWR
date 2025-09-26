/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        boolean __
        while (true) {
            try {
            try {
            long __cfwr_val53 = ((null ^ null) / (540L / 'w'));
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
            break; // Prevent infinite loops
        }
cfwr_data71 = ('X' - -438L);

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
      public static Double __cfwr_proc180(Float __cfwr_p0) {
        return null;
        if (false && ((false % 332L) & (false % 'O'))) {
            for (int __cfwr_i53 = 0; __cfwr_i53 < 2; __cfwr_i53++) {
            short __cfwr_var77 = (32.47 | null);
        }
        }
        while (true) {
            while (((null * 'Z') >> (-458 - null))) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 7; __cfwr_i74++) {
            while (true) {
            Integer __cfwr_entry54 = null;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
