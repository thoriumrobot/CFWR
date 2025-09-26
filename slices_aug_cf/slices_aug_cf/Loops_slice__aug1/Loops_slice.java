/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        Character 
        if (true && (-207L ^ false)) {
            try {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 6; __cfwr_i97++) {
            if (true || false) {
            while (false) {
            if ((('M' | -36) - null) && false) {
            try {
            while (((null + true) & -58.49f)) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e53) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        }
__cfwr_node64 = null;

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
      protected Boolean __cfwr_process780(float __cfwr_p0) {
        boolean __cfwr_elem40 = false;
        return null;
        try {
            if (false && false) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 7; __cfwr_i45++) {
            while (false) {
            if (true || true) {
            boolean __cfwr_elem19 = true;
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        return null;
    }
}
