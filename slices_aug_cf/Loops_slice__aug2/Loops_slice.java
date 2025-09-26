/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        char __cfw
        for (int __cfwr_i84 = 0; __cfwr_i84 < 10; __cfwr_i84++) {
            if (true && true) {
            Character __cfwr_node35 = null;
        }
        }
r_temp36 = 'Y';

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
      protected Double __cfwr_func269(Double __cfwr_p0, boolean __cfwr_p1) {
        while (true) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 1; __cfwr_i45++) {
            Long __cfwr_temp45 = null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i66 = 0; __cfwr_i66 < 1; __cfwr_i66++) {
            while (true) {
            Object __cfwr_entry60 = null;
            break; // Prevent infinite loops
        }
        }
        if (false && false) {
            return ('L' << null);
        }
        return null;
    }
    private Object __cfwr_func534() {
        return 716;
        try {
            Double __cfwr_entry86 = null;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        try {
            while ((-40.98 >> null)) {
            for (int __cfwr_i18 = 0; __cfwr_i18 < 2; __cfwr_i18++) {
            int __cfwr_result29 = 949;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        try {
            return null;
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        return null;
    }
    private int __cfwr_aux41() {
        try {
            if ((12.05 & true) && (18.73 + 897L)) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        return 888;
    }
}
