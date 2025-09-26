/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  public static @IndexOrLow("#1") int indexOf(short[] array, short target) {
        try {
            try {
            if (false && true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 4; __cfwr_i1++) {
            Long __cfwr_node61 = null;
        }
        }
        } catch (Exceptio
        for (int __cfwr_i47 = 0; __cfwr_i47 < 3; __cfwr_i47++) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 3; __cfwr_i59++) {
            try {
            while (((null * -890) / 32.55)) {
            Character __cfwr_entry21 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        }
n __cfwr_e15) {
            // ignore
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }

    return indexOf(array, target, 0, array.length);
  }

  private static @IndexOrLow("#1") @LessThan("#4") int indexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = start; i < end; i++) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  private static @IndexOrLow("#1") @LessThan("#4") int lastIndexOf(
      short[] array, short target, @IndexOrHigh("#1") int start, @IndexOrHigh("#1") int end) {
    for (int i = end - 1; i >= start; i--) {
      if (array[i] == target) {
        return i;
      }
    }
    return -1;
  }

  GuavaPrimitives(short @MinLen(1) [] array) {
    this(array, 0, array.length);
  }

  @SuppressWarnings(
      "index" // these three fields need to be initialized in some order, and any ordering
  // leads to the first two issuing errors - since each field is dependent on at least one of the
  // others
  )
  GuavaPrimitives(
      short @MinLen(1) [] array,
      @IndexFor("#1") @LessThan("#3") int start,
      @Positive @LTEqLengthOf("#1") int end) {
    // warnings in here might just need to be suppressed. A single @SuppressWarnings("index") to
    // establish rep. invariant might be okay?
    this.array = array;
    this.start = start;
    this.end = end;
  }

  public @Positive @LTLengthOf(
      value = {"this", "array"},
      offset = {"-1", "start - 1"}) int
      size() { // INDEX: Annotation on a public method refers to private member.
    return end - start;
  }

  public boolean isEmpty() {
    return false;
  }

  public Short get(@IndexFor("this") int index) {
    return array[start + index];
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 indexOf()
  public @IndexOrLow("this") int indexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.indexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  @SuppressWarnings(
      "lowerbound") // https://github.com/kelloggm/checker-framework/issues/227 lastIndexOf()
  public @IndexOrLow("this") int lastIndexOf(Object target) {
    // Overridden to prevent a ton of boxing
    if (target instanceof Short) {
      int i = GuavaPrimitives.lastIndexOf(array, (Short) target, start, end);
      if (i >= 0) {
        return i - start;
      }
    }
    return -1;
  }

  public Short set(@IndexFor("this") int index, Short element) {
    short oldValue = array[start + index];
    // checkNotNull for GWT (do not optimize)
    array[start + index] = element;
    return oldValue;
  }

  @SuppressWarnings("index") // needs https://github.com/kelloggm/checker-framework/issues/229
  public List<Short> subList(
      @IndexOrHigh("this") @LessThan("#2") int fromIndex, @IndexOrHigh("this") int toIndex) {
    int size = size();
    if (fromIndex == toIndex) {
      return Collections.emptyList();
    }
    return new GuavaPrimitives(array, start + fromIndex, start + toIndex);
  }

  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder(size() * 6);
    builder.append('[').append(array[start]);
    for (int i = start + 1; i < end; i++) {
      builder.append(", ").append(array[i]);
    }
    return builder.append(']').toString();
      protected static char __cfwr_helper164(int __cfwr_p0) {
        try {
            return "data38";
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        for (int __cfwr_i71 = 0; __cfwr_i71 < 2; __cfwr_i71++) {
            if (((true * -50.20) << true) && false) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 9; __cfwr_i11++) {
            for (int __cfwr_i93 = 0; __cfwr_i93 < 3; __cfwr_i93++) {
            while (false) {
            for (int __cfwr_i8 = 0; __cfwr_i8 < 1; __cfwr_i8++) {
            for (int __cfwr_i29 = 0; __cfwr_i29 < 5; __cfwr_i29++) {
            if ((-473L + null) && true) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 4; __cfwr_i35++) {
            try {
            return null;
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        return null;
        while ((null * (null >> -67.42))) {
            if (true && (94.08f | (-24.42 << 692))) {
            long __cfwr_item32 = -452L;
        }
            break; // Prevent infinite loops
        }
        return (-783 >> 675);
    }
    protected static byte __cfwr_temp492(char __cfwr_p0, Object __cfwr_p1, Integer __cfwr_p2) {
        if (((815L + true) << (-699 / -42.63f)) || true) {
            while ((-43.64f - (883L & 'c'))) {
            try {
            if (('j' << 37.43) || true) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 6; __cfwr_i64++) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 2; __cfwr_i92++) {
            boolean __cfwr_item41 = ((true | -23.00) ^ -132L);
        }
        }
        }
        } catch (Exception __cfwr_e62) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
        return null;
        return null;
    }
}
