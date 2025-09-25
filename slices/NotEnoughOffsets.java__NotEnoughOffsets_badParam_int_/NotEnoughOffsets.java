import org.checkerframework.checker.index.qual.LTLengthOf;

public class NotEnoughOffsets {

    void badParam(@LTLengthOf(value = { "a", "b" }, offset = { "c" }) int x) {
    }
}
