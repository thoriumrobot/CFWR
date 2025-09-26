/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public void test1a(int[] a, @LTLengthOf("#1") int offset, @LTLengthOf("#1") int offset2) {
        Character __cfwr_val96 = null;

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
      Character __cfwr_proc773(String __cfwr_p0, String __cfwr_p1) {
        byte __cfwr_data59 = ('W' << 92.81f);
        while (true) {
            if (false && ((887L + -579) & (712L << 36.37f))) {
            Character __cfwr_elem97 = null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    private float __cfwr_proc405(double __cfwr_p0) {
        if (false || true) {
            Double __cfwr_item92 = null;
        }
        return (('D' | -398L) / 71.71);
    }
}
